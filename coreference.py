# coding=utf-8

from __future__ import absolute_import, division, print_function

import json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import cmp_to_key
from tqdm import tqdm, trange

# 项目模块
import utils
from utils import to_cuda
import conll
import metrics
from bert import tokenization, modeling
from bert.tokenization import BertTokenizer
from bert.modeling import BertPreTrainedModel
from bert.modeling import BertModel


class Squeezer(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return torch.squeeze(input, dim=self.dim)


class Score(nn.Module):
    """计算得分"""
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.3):
        super(Score, self).__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input):
        output = self.score(input)
        return output


class CorefModel(BertPreTrainedModel):

    def __init__(self, config, coref_task_config):
        super(CorefModel, self).__init__(config)

        self.config = coref_task_config
        self.max_segment_len = self.config['max_segment_len']
        self.max_span_width = self.config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(self.config["genres"])}
        self.subtoken_maps = {}
        self.gold = {}
        self.eval_data = None
        self.bert_config = modeling.BertConfig.from_json_file(self.config["bert_config_file"])
        self.tokenizer = BertTokenizer.from_pretrained(self.config['vocab_file'], do_lower_case=True)
        self.bert = BertModel(config=self.bert_config)
        self.dropout = nn.Dropout(self.config["dropout_rate"])
        self.emb_dim = self.bert_config.hidden_size*2 + int(self.config["use_features"])*20 + int(self.config["model_heads"])*self.bert_config.hidden_size
        self.slow_antecedent_dim = self.emb_dim*3 + int(self.config["use_metadata"])*40 + int(self.config["use_features"])*20 + int(self.config['use_segment_distance'])*20

        # span 长度 Embedding
        if self.config["use_features"]:
            self.span_width_embedding = nn.Embedding(
                                        num_embeddings=self.config["max_span_width"],
                                        embedding_dim=self.config["feature_size"])
        # span head Embedding(ok)
        if self.config["model_heads"]:
            print("------加入span head 信息------")
            self.masked_mention_score = nn.Sequential(
                                                nn.Linear(self.bert_config.hidden_size, 1),
                                                Squeezer(dim=1))

        # 计算指代得分,两层前向神经网络(ok)
        self.mention_scores = Score(self.emb_dim, self.config["ffnn_size"])

        # prior_width_embedding
        if self.config['use_prior']:
            self.span_width_prior_embeddings = nn.Embedding(
                                        num_embeddings=self.config["max_span_width"],
                                        embedding_dim=self.config["feature_size"])

            # 计算长度得分，两层前向神经网络
            self.width_scores = Score(self.config["feature_size"], self.config["ffnn_size"])

        # doc类别Embedding[7,20]
        self.genres_embedding = nn.Embedding(
                                        num_embeddings=len(self.genres),
                                        embedding_dim=self.config["feature_size"])

        # 前c个前指的得分  一个分类器 + dropout
        self.fast_antecedent_scores = nn.Sequential(
                                    nn.Linear(self.emb_dim, self.emb_dim),
                                    nn.Dropout(self.config["dropout_rate"]))
        # 前指距离embedding
        if self.config['use_prior']:
            self.antecedent_distance_embedding = nn.Embedding(
                                            num_embeddings=10,
                                            embedding_dim=self.config["feature_size"])

            self.antecedent_distance_linear = nn.Linear(self.config["feature_size"], 1)

        if self.config["use_metadata"]:
            # [2,20]
            self.same_speaker_embedding = nn.Embedding(
                                                num_embeddings=2,
                                                embedding_dim=self.config["feature_size"])
        if self.config["use_features"]:
            self.antecedent_offset_embedding = nn.Embedding(
                                            num_embeddings=10,
                                            embedding_dim=self.config["feature_size"])
        if self.config['use_segment_distance']:
            self.segment_distance_embedding = nn.Embedding(
                                            num_embeddings=self.config['max_training_sentences'],
                                            embedding_dim=self.config["feature_size"])

        # 三维的输入 ffnn 两层前向神经网络
        if self.config['fine_grained']:
            self.slow_antecedent_scores = nn.Sequential(
                nn.Linear(self.slow_antecedent_dim, self.config["ffnn_size"]),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config["dropout_rate"]),
                nn.Linear(self.config["ffnn_size"], 1),
                Squeezer(dim=-1)
            )

            # 分类器 + sigmoid
            self.coref_layer_linear = nn.Sequential(
                            nn.Linear(self.emb_dim*2, self.emb_dim),
                            nn.Sigmoid()
            )

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map):
        # bert_encoder最后一层输出
        emb_mention_doc, _ = self.bert(input_ids=input_ids, attention_mask=input_mask, output_all_encoded_layers=False)  # [batch_size, seg_len, hidden_size]

        mention_doc = self.flatten_emb_by_sentence(emb_mention_doc, input_mask)    # [batch_size*seg_len, hidden_size]
        num_words = torch.tensor(mention_doc.shape[0])    # [batch_size*seg_len]
        # 根据最大子串长度，获得候选子串
        flattened_sentence_indices = sentence_map     # num_word
        candidate_starts = torch.arange(num_words).view(-1, 1).repeat(1, self.max_span_width)   # [num_words_len, max_span_width]
        candidate_ends = candidate_starts + torch.arange(self.max_span_width).view(1, -1)   # [num_words_len, max_span_width]
        # 句子开始、结束索引
        candidate_start_sentence_indices = flattened_sentence_indices[candidate_starts]  # [num_words_len, max_span_width]
        candidate_end_sentence_indices = flattened_sentence_indices[torch.clamp(candidate_ends, max=num_words-1)] # [num_words_len, max_span_width]
        # torch.min(candidate_ends, 225*torch.ones([candidate_ends.shape[0], candidate_ends.shape[1]]).long())

        candidate_mask = to_cuda((candidate_ends < num_words)) & to_cuda(torch.eq(candidate_start_sentence_indices, candidate_end_sentence_indices))  # [num_words_len, max_span_width]
        flattened_candidate_mask = candidate_mask.view(-1)                     # [num_words * max_span_width]

        candidate_starts = candidate_starts.view(-1)[flattened_candidate_mask]    # [num_candidates]
        candidate_ends = candidate_ends.view(-1)[flattened_candidate_mask]        # [num_candidates]
        # 候选簇
        if is_training:
            candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends,
                                                          gold_starts, gold_ends, cluster_ids)       # [num_candidates]
        # 输入的Embedding
        candidate_span_emb = self.get_span_emb(mention_doc, candidate_starts, candidate_ends)
        # 指代得分
        candidate_mention_scores = self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)
        candidate_mention_scores = candidate_mention_scores.squeeze(1)              # [num_candidates]

        # 裁剪得分低的指代
        max_vlaue = torch.floor(num_words.type(torch.float)) * self.config["top_span_ratio"]
        k = torch.clamp(torch.tensor(3900), max=max_vlaue.int())
        c = torch.clamp(torch.tensor(self.config["max_top_antecedents"]), max=k)

        top_span_indices = self.extract_top_spans(candidate_mention_scores, candidate_starts, candidate_ends, k)
        top_span_indices = top_span_indices.type(torch.int64)

        # 裁剪后的短语开始结束索引，embedding，簇，得分
        top_span_starts = candidate_starts[top_span_indices]  # [k]
        top_span_ends = candidate_ends[top_span_indices]  # [k]
        top_span_emb = candidate_span_emb[top_span_indices]  # [k, emb]

        if is_training:
            top_span_cluster_ids = candidate_cluster_ids[top_span_indices]  # [k]

        top_span_mention_scores = candidate_mention_scores[top_span_indices]  # [k]
        genre_emb = self.genres_embedding(genre)  # [20,]

        # 加入说话人信息
        if self.config['use_metadata']:
            speaker_ids = self.flatten_emb_by_sentence(speaker_ids, input_mask)
            top_span_speaker_ids = speaker_ids[top_span_starts]
        else:
            top_span_speaker_ids = None

        dummy_scores = to_cuda(torch.zeros(k, 1))  # [k,1]
        # 得到top-c个前指
        top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = \
            self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)

        num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
        word_segments = torch.arange(num_segs).view(-1, 1).repeat(1, seg_len)
        bool_inputmask = input_mask == 1
        # bool_inputmask = torch.tensor(bool_inputmask)
        flat_word_segments = word_segments.view(-1)[bool_inputmask.view(-1)]

        mention_segments = flat_word_segments[top_span_starts].view(-1, 1)
        antecedent_segments = flat_word_segments[top_span_starts[top_antecedents]]

        if self.config['use_segment_distance']:
            segment_distance = torch.clamp((mention_segments - antecedent_segments),
                                           0, (self.config['max_training_sentences'] - 1))
        else:
            segment_distance = None
        ##计算top_c个前指得分
        # 二阶
        if self.config['fine_grained']:
            for i in range(self.config["coref_depth"]):

                top_antecedent_emb = top_span_emb[top_antecedents]  #[k,c,emb]
                slow_antecedent_scores = self.get_slow_antecedent_scores(top_span_emb, top_antecedents,
                                                                         top_antecedent_emb, top_antecedent_offsets,
                                                                         top_span_speaker_ids, genre_emb,
                                                                         segment_distance)

                top_antecedent_scores = top_fast_antecedent_scores + slow_antecedent_scores
                top_antecedent_weights = F.softmax(torch.cat((dummy_scores, top_antecedent_scores), dim=1), dim=-1)  # [k, c + 1]

                top_antecedent_emb = torch.cat((top_span_emb.unsqueeze(1), top_antecedent_emb), dim=1)  # [k, c + 1, emb]
                attended_span_emb = torch.sum(top_antecedent_weights.unsqueeze(2) * top_antecedent_emb, 1)  # [k, emb]

                cat_span_emb = torch.cat((top_span_emb, attended_span_emb), dim=1)

                f = self.coref_layer_linear(cat_span_emb)
                top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]
        # 一阶
        else:
            top_antecedent_scores = top_fast_antecedent_scores
        top_antecedent_scores = torch.cat((dummy_scores, top_antecedent_scores), dim=1)   # [k, c + 1]
        # 非训练阶段省去计算loss
        if not is_training:
            loss = torch.tensor(0)
            return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                    top_antecedents, top_antecedent_scores], loss

        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedents]  # [k, c]
        top_antecedent_cluster_ids += to_cuda(torch.log(top_antecedents_mask.float()))  # [k, c]

        same_cluster_indicator = torch.eq(top_antecedent_cluster_ids, top_span_cluster_ids.view(-1, 1))
        non_dummy_indicator = (top_span_cluster_ids > 0).view(-1, 1)  # [k, 1]
        pairwise_labels = same_cluster_indicator & non_dummy_indicator  # [k, c]

        dummy_labels = ~ (pairwise_labels.any(dim=1, keepdim=True))   # [k, 1]
        top_antecedent_labels = torch.cat((dummy_labels, pairwise_labels), dim=1)  # [k, c+1]
        # loss函数
        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)
        loss = torch.sum(loss)

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores], loss

    def get_train_example(self):
        with open(self.config["train_path"], encoding="utf-8") as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]
        return examples

    def get_eval_example(self):
        with open(self.config["eval_path"], encoding="utf-8") as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]
        return examples

    def get_test_example(self):
        with open(self.config["test_path"], encoding="utf-8") as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]
        return examples

    def tensorize_example(self, example, is_training):
        """样例处理"""
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in utils.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))  # 计算每个簇的次数
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = example["speakers"]
        # assert num_words == len(speakers), (num_words, len(speakers))
        speaker_dict = self.get_speaker_dict(utils.flatten(speakers))
        sentence_map = example['sentence_map']

        max_sentence_length = self.config["max_segment_len"]
        text_len = np.array([len(s) for s in sentences])

        input_ids, input_mask, speaker_ids = [], [], []
        for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]
            while len(sent_input_ids) < max_sentence_length:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)

            input_ids.append(sent_input_ids)
            speaker_ids.append(sent_speaker_ids)
            input_mask.append(sent_input_mask)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        doc_key = example["doc_key"]
        self.subtoken_maps[doc_key] = example.get("subtoken_map", None)
        self.gold[doc_key] = example["clusters"]
        genre = self.genres.get(doc_key[:2], 0)

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        example_tensors = (input_ids, input_mask, text_len, speaker_ids, genre, is_training,
                           gold_starts, gold_ends, cluster_ids, sentence_map)

        if is_training and len(sentences) > self.config["max_training_sentences"]:
            if self.config['single_example']:
                return self.truncate_example(*example_tensors)
            else:
                offsets = range(self.config['max_training_sentences'], len(sentences),
                                self.config['max_training_sentences'])
                tensor_list = [self.truncate_example(*(example_tensors + (offset,))) for offset in offsets]
                return tensor_list
        else:
            return example_tensors

    def get_speaker_dict(self, speakers):
        """返回说话者对应的字典"""
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for s in speakers:
            if s not in speaker_dict and len(speaker_dict) < self.config['max_num_speakers']:
                speaker_dict[s] = len(speaker_dict)
        return speaker_dict

    def tensorize_mentions(self, mentions):
        """找到指代的起始"""
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def truncate_example(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends,
                         cluster_ids, sentence_map, sentence_offset=None):
        """截断句长"""
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences) if sentence_offset is None \
                                                                                    else sentence_offset
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
        speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        """根据mask展平embedding"""
        num_sentences = emb.shape[0]
        max_sentence_length = emb.shape[1]

        emb_rank = emb.dim()
        if emb_rank == 2:
            flattened_emb = emb.view(num_sentences*max_sentence_length)
        elif emb_rank == 3:
            flattened_emb = emb.view(num_sentences*max_sentence_length, emb.shape[2])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        mask = text_len_mask.view(num_sentences*max_sentence_length) == 1
        return flattened_emb[mask]

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = torch.eq(to_cuda(labeled_starts.view(-1, 1)), to_cuda(candidate_starts.view(1, -1)))  # [num_labeled, num_candidates]
        same_end = torch.eq(to_cuda(labeled_ends.view(-1, 1)), to_cuda(candidate_ends.view(1, -1)))  # [num_labeled, num_candidates]
        same_span = same_start & same_end                                       # [num_labeled, num_candidates]
        candidate_labels = torch.matmul(to_cuda(labels.view(1, -1).float()), same_span.float())  # [1, num_candidates]
        candidate_labels = candidate_labels.squeeze(0)  # [num_candidates]
        return candidate_labels

    def get_span_emb(self, context_outputs, span_starts, span_ends):
        """得到span_embedding([span start, span end, span width embedding, span head embedding])"""
        span_emb_list = []
        # span start
        span_start_emb = context_outputs[span_starts]  # [num_candidates ,hidden_size]
        span_emb_list.append(span_start_emb)
        # span end
        span_end_emd = context_outputs[span_ends]
        span_emb_list.append(span_end_emd)

        span_width = 1 + span_ends - span_starts  # [num_candidates]

        # span width embedding
        if self.config["use_features"]:
            span_width_index = span_width - 1    # [num_candidates]
            span_width_emb = self.span_width_embedding(to_cuda(span_width_index))  # [num_candidates, self.config["feature_size"]]
            span_width_emb = self.dropout(span_width_emb)
            span_emb_list.append(span_width_emb)   # [num_candidates, 20]

        # span head embedding
        if self.config["model_heads"]:
            mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
            head_attn_emb = torch.matmul(mention_word_scores, context_outputs)  # [K, T]
            span_emb_list.append(head_attn_emb)

        span_emb = torch.cat(span_emb_list, 1)
        return span_emb

    def get_mention_scores(self, span_emb, span_starts, span_ends):
        """计算指代得分(span embedding 经过两层前向神经网络)
        """
        span_scores = self.mention_scores(span_emb)

        if self.config['use_prior']:
            span_width_emb = self.span_width_prior_embeddings.weight  # [30,20]
            span_width_index = span_ends - span_starts

            width_scores = self.width_scores(span_width_emb)
            width_scores = width_scores[span_width_index]
            span_scores += width_scores

        return span_scores


    @staticmethod
    def extract_top_spans(span_scores, cand_start_idxes, cand_end_idxes, top_span_num):
        """获得前k个短语"""

        sorted_span_idxes = torch.argsort(span_scores, descending=True).tolist()

        top_span_idxes = []
        end_idx_to_min_start_dix, start_idx_to_max_end_idx = {}, {}
        selected_span_num = 0

        for span_idx in sorted_span_idxes:
            crossed = False
            start_idx = cand_start_idxes[span_idx]
            end_idx = cand_end_idxes[span_idx]

            if end_idx == start_idx_to_max_end_idx.get(start_idx, -1):
                continue

            for j in range(start_idx, end_idx + 1):
                if j in start_idx_to_max_end_idx and j > start_idx and start_idx_to_max_end_idx[j] > end_idx:
                    crossed = True
                    break

                if j in end_idx_to_min_start_dix and j < end_idx and end_idx_to_min_start_dix[j] < start_idx:
                    crossed = True
                    break

            if not crossed:
                top_span_idxes.append(span_idx)
                selected_span_num += 1

                if start_idx not in start_idx_to_max_end_idx or end_idx > start_idx_to_max_end_idx[start_idx]:
                    start_idx_to_max_end_idx[start_idx] = end_idx

                if end_idx not in end_idx_to_min_start_dix or start_idx < end_idx_to_min_start_dix[end_idx]:
                    end_idx_to_min_start_dix[end_idx] = start_idx

            if selected_span_num == top_span_num:
                break

        def compare_span_idxes(i1, i2):
            if cand_start_idxes[i1] < cand_start_idxes[i2]:
                return -1
            elif cand_start_idxes[i1] > cand_start_idxes[i2]:
                return 1
            elif cand_end_idxes[i1] < cand_end_idxes[i2]:
                return -1
            elif cand_end_idxes[i1] > cand_end_idxes[i2]:
                return 1
            else:
                return 0

        top_span_idxes.sort(key=cmp_to_key(compare_span_idxes))

        return (torch.Tensor(top_span_idxes) + torch.tensor(top_span_idxes[0]) * (top_span_num - selected_span_num))

    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
        """计算前c个得分"""
        k = top_span_emb.shape[0]
        top_span_range = torch.arange(k)   # [k]

        antecedent_offsets = top_span_range.view(-1, 1) - top_span_range.view(1, -1)  # [k, k]
        antecedents_mask = antecedent_offsets >= 1  # [k, k]

        fast_antecedent_scores = top_span_mention_scores.view(-1, 1) + top_span_mention_scores.view(1, -1)  # [k, k]
        fast_antecedent_scores += torch.log(to_cuda(antecedents_mask.float()))
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb)

        if self.config['use_prior']:
            antecedent_distance_buckets = self.get_offset_bucket_idxes_batch(antecedent_offsets)

            antecedent_distance_emb = self.antecedent_distance_embedding.weight
            antecedent_distance_emb = self.dropout(antecedent_distance_emb)
            distance_scores = self.antecedent_distance_linear(antecedent_distance_emb)

            antecedent_distance_scores = distance_scores.squeeze(1)[antecedent_distance_buckets]

            fast_antecedent_scores += antecedent_distance_scores

        _, top_antecedents = torch.topk(fast_antecedent_scores, c, sorted=False)

        top_antecedents_mask = self.batch_gather(antecedents_mask, top_antecedents)  # [k, c]
        top_fast_antecedent_scores = self.batch_gather(fast_antecedent_scores, top_antecedents)  # [k, c]
        top_antecedent_offsets = self.batch_gather(antecedent_offsets, top_antecedents)

        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def get_fast_antecedent_scores(self, top_span_emb):
        """计算前K个前指的得分"""
        source_top_span_emb = self.fast_antecedent_scores(top_span_emb)
        target_top_span_emb = self.dropout(top_span_emb)

        return torch.matmul(source_top_span_emb, target_top_span_emb.t())

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   top_span_speaker_ids, genre_emb, segment_distance=None):
        k = top_span_emb.shape[0]
        c = top_antecedents.shape[1]

        feature_emb_list = []

        # 说话者、类别信息
        if self.config["use_metadata"]:
            top_antecedent_speaker_ids = top_span_speaker_ids[top_antecedents]
            same_speaker = torch.eq(top_span_speaker_ids.view(-1, 1), top_antecedent_speaker_ids)
            speaker_pair_emb = self.same_speaker_embedding(same_speaker.type(torch.int64))  # [k, c, emb20]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = genre_emb.view(1, -1).view(1, -1).repeat(k, c, 1)  # [k, c, emb20]
            feature_emb_list.append(tiled_genre_emb)
        #
        if self.config["use_features"]:
            antecedent_distance_buckets = to_cuda(self.get_offset_bucket_idxes_batch(top_antecedent_offsets))
            antecedent_distance_emb = self.antecedent_offset_embedding(antecedent_distance_buckets)  #[k, c, emb20]
            feature_emb_list.append(antecedent_distance_emb)  #[k, c, emb20]
        #
        if segment_distance is not None:
            segment_distance_emb = self.segment_distance_embedding(to_cuda(segment_distance))  #[k,emb]
            feature_emb_list.append(segment_distance_emb)

        feature_emb = torch.cat(feature_emb_list, 2)  # [k, c, emb80 每个特征20]
        feature_emb = self.dropout(feature_emb)

        target_emb = top_span_emb.unsqueeze(1)  # [k, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [k, c, emb]
        target_emb = target_emb.repeat(1, c, 1)  # [k, c, emb]
        # 三维
        pair_emb = torch.cat((target_emb, top_antecedent_emb, similarity_emb, feature_emb), 2)  # [k, c, emb]

        slow_antecedent_scores = self.slow_antecedent_scores(pair_emb)  # [k, c]

        return slow_antecedent_scores

    def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
        num_words = encoded_doc.shape[0]
        num_c = span_starts.shape[0]

        doc_range = torch.arange(num_words).view(1, -1).repeat(num_c, 1)
        mention_mask = (doc_range >= (span_starts.view(-1, 1))) & (doc_range <= span_ends.view(-1, 1))

        word_attn = self.masked_mention_score(encoded_doc)
        mention_word_attn = F.softmax(torch.log(to_cuda(mention_mask.float())) + to_cuda(word_attn.view(1, -1)), dim=-1)

        return mention_word_attn

    def get_offset_bucket_idxes_batch(self, offsets_batch):
        """
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        log_space_idxes_batch = (torch.log(offsets_batch.float()) / math.log(2)).floor().long() + 3

        identity_mask_batch = (offsets_batch <= 4).long()

        return torch.clamp(
            identity_mask_batch * offsets_batch + (1 - identity_mask_batch) * log_space_idxes_batch, min=0, max=9)

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float())
        marginalized_gold_scores = torch.logsumexp(gold_scores, dim=1)  # [k]
        log_norm = torch.logsumexp(antecedent_scores, dim=1)  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    def batch_gather(self, emb, indices):
        batch_size = emb.shape[0]
        seqlen = emb.shape[1]
        if len(emb.shape) > 2:
            emb_size = emb.shape[2]
        else:
            emb_size = 1
        flattened_emb = emb.view(batch_size * seqlen, emb_size)
        offset = to_cuda((torch.arange(batch_size) * seqlen).view(-1, 1))
        gathered = flattened_emb[indices + offset]

        if len(emb.shape) == 2:
            gathered = torch.squeeze(gathered, 2)
        return gathered

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        """获得预测的前值簇"""
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
        """获得预测的共指簇"""
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index, (i, predicted_index)
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                               predicted_antecedents)
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def load_eval_data(self):
        if self.eval_data is None:
            with open(self.config["eval_path"]) as f:
                self.eval_data = [json.loads(jsonline) for jsonline in f.readlines()]

            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def evaluate(self, model, device, official_stdout=False, keys=None, eval_mode=False):
        self.load_eval_data()

        coref_predictions = {}
        coref_evaluator = metrics.CorefEvaluator()
        doc_keys = []

        with torch.no_grad():
            for example_num, example in enumerate(tqdm(self.eval_data, desc="Eval_Examples")):
                tensorized_example = model.tensorize_example(example, is_training=False)

                input_ids = torch.from_numpy(tensorized_example[0]).long().to(device)
                input_mask = torch.from_numpy(tensorized_example[1]).long().to(device)
                text_len = torch.from_numpy(tensorized_example[2]).long().to(device)
                speaker_ids = torch.from_numpy(tensorized_example[3]).long().to(device)
                genre = torch.tensor(tensorized_example[4]).long().to(device)
                is_training = tensorized_example[5]
                gold_starts = torch.from_numpy(tensorized_example[6]).long().to(device)
                gold_ends = torch.from_numpy(tensorized_example[7]).long().to(device)
                cluster_ids = torch.from_numpy(tensorized_example[8]).long().to(device)
                sentence_map = torch.Tensor(tensorized_example[9]).long().to(device)

                if keys is not None and example['doc_key'] not in keys:
                    continue
                doc_keys.append(example['doc_key'])

                (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                 top_antecedents, top_antecedent_scores), loss = model(input_ids, input_mask, text_len, speaker_ids,
                                                genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map)

                predicted_antecedents = self.get_predicted_antecedents(top_antecedents.cpu(), top_antecedent_scores.cpu())
                coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends,
                                                                            predicted_antecedents, example["clusters"],
                                                                            coref_evaluator)

        summary_dict = {}
        if eval_mode:
            conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, self.subtoken_maps,
                                                 official_stdout)
            average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            summary_dict["Average F1 (conll)"] = average_f1
            print("Average F1 (conll): {:.2f}%".format(average_f1))

        p, r, f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        return summary_dict, f
