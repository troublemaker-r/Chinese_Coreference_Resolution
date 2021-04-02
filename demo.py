#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
import logging
import random
import torch
import torch.optim as optim
from tqdm import tqdm, trange

from bert.tokenization import BertTokenizer
import utils
from coreference import CorefModel
import conll
import metrics

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_coref(config):
    """
    指代消解模型训练
    :param config: 配置参数
    :return: None
    """
    model = CorefModel.from_pretrained(config["pretrained_model"], coref_task_config=config)
    print(model)
    model.to(device)

    examples = model.get_train_example()
    train_steps = config["num_epochs"] * config["num_docs"]

    param_optimizer = list(model.named_parameters())
    print("需要学习的参数：{}".format(len(param_optimizer)))

    bert_params = list(map(id, model.bert.parameters()))
    task_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    # 优化器
    optimizer = optim.Adam([
        {'params': task_params},
        {'params': model.bert.parameters(), 'lr': config['bert_learning_rate']}],
        lr=config['task_learning_rate'],
        eps=config['adam_eps'])

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=int(train_steps * 0.1))

    logger.info("********** Running training ****************")
    logger.info("  Num train examples = %d", len(examples))
    logger.info("  Num epoch = %d", config["num_epochs"])
    logger.info("  Num train step = %d", train_steps)

    fh = logging.FileHandler(os.path.join(config["data_dir"], 'train.log'), mode="w")
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    model.train()
    global_step = 0
    start_time = time.time()
    accumulated_loss = 0.0

    for _ in trange(int(config["num_epochs"]), desc="Epoch"):
        random.shuffle(examples)
        for step, example in enumerate(tqdm(examples, desc="Train_Examples")):
            tensorized_example = model.tensorize_example(example, is_training=True)

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

            predictions, loss = model(input_ids, input_mask, text_len, speaker_ids, genre, is_training,
                                      gold_starts, gold_ends, cluster_ids, sentence_map)

            accumulated_loss += loss.item()
            if global_step % report_frequency == 0:
                total_time = time.time() - start_time
                steps_per_second = global_step / total_time
                average_loss = accumulated_loss / report_frequency
                print("\n")
                logger.info("step:{} | loss: {} | step/s: {}".format(global_step, average_loss, steps_per_second))
                accumulated_loss = 0.0
            # 验证集验证
            if global_step % eval_frequency == 0 and global_step != 0:
                utils.save_model(model, config["model_save_path"])
                torch.cuda.empty_cache()
                eval_model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
                eval_model.to(device)
                eval_model.eval()
                try:
                    eval_model.evaluate(eval_model, device, official_stdout=True, eval_mode=True)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                except AttributeError as exception:
                    print("Found too many repeated mentions (> 10) in the response, so refusing to score")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        scheduler.step()

    utils.save_model(model, config["model_save_path"])
    print("*****************************训练完成，已保存模型****************************************")
    torch.cuda.empty_cache()


def eval_coref(config):
    """
    指代消解模型验证
    :param config: 配置参数
    :return: None
    """
    model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
    model.to(device)

    examples = model.get_eval_example()

    logger.info("********** Running Eval ****************")
    logger.info("  Num dev examples = %d", len(examples))

    model.eval()
    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    doc_keys = []
    keys = None
    with torch.no_grad():
        for example_num, example in enumerate(tqdm(examples, desc="Eval_Examples")):
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

            predicted_antecedents = model.get_predicted_antecedents(top_antecedents.cpu(), top_antecedent_scores.cpu())
            coref_predictions[example["doc_key"]] = model.evaluate_coref(top_span_starts, top_span_ends,
                                                                        predicted_antecedents, example["clusters"],
                                                                        coref_evaluator)
    official_stdout = True
    eval_mode = True
    summary_dict = {}
    if eval_mode:
        conll_results = conll.evaluate_conll(config["conll_eval_path"], coref_predictions,
                                             model.subtoken_maps, official_stdout)
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


def test_coref(config):
    """
    指代消解模型预测
    :param config: 配置参数
    :return: None 
    """
    model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
    model.to(device)

    output_filename = config["test_output_path"]
    examples = model.get_test_example()

    logger.info("********** Running Test ****************")
    logger.info("  Num test examples = %d", len(examples))

    model.eval()
    with open(output_filename, 'w', encoding="utf-8") as output_file:
        with torch.no_grad():
            for example_num, example in enumerate(tqdm(examples, desc="Test_Examples")):
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

                (_, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores), _ = \
                                        model(input_ids, input_mask, text_len, speaker_ids, genre,
                                              is_training, gold_starts, gold_ends,
                                              cluster_ids, sentence_map)

                predicted_antecedents = model.get_predicted_antecedents(top_antecedents.cpu(), top_antecedent_scores.cpu())
                example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                                predicted_antecedents)
                # 将句中索引——>文字
                example_sentence = utils.flatten(example["sentences"])
                predicted_list = []
                for same_entity in example["predicted_clusters"]:
                    same_entity_list = []
                    num_same_entity = len(same_entity)
                    for index in range(num_same_entity):
                        entity_name = ''.join(example_sentence[same_entity[index][0]: same_entity[index][1]+1])
                        same_entity_list.append(entity_name)
                    predicted_list.append(same_entity_list)
                    same_entity_list = []   # 清空list

                example["predicted_idx2entity"] = predicted_list
                example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
                example['head_scores'] = []

                output_file.write(json.dumps(example, ensure_ascii=False))
                output_file.write("\n")
                if example_num % 100 == 0:
                    print('\n')
                    print("写入 {} examples.".format(example_num + 1))


def online_test_coref(config, input_text):
    """
    输入一段文本，进行指代消解任务
    :param config: 配置参数
    :return: None
    """

    def create_example(text):
        """将文字转为模型需要的样例格式"""
        sentences = [['[CLS]'] + tokenizer.tokenize_not_UNK(text) + ['[SEP]']]
        sentence_map = [0] * len(sentences[0])
        speakers = [["-" for _ in sentence] for sentence in sentences]
        subtoken_map = [i for i in range(len(sentences[0]))]
        return {
            "doc_key": "bn",
            "clusters": [],
            "sentences": sentences,
            "speakers": speakers,
            'sentence_map': sentence_map,
            'subtoken_map': subtoken_map
        }

    tokenizer = BertTokenizer.from_pretrained(config['vocab_file'], do_lower_case=True)
    online_coref_output_file = config['online_output_path']

    example = create_example(input_text)

    model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
    model.to(device)

    model.eval()
    with open(online_coref_output_file, 'w', encoding="utf-8") as output_file:

        with torch.no_grad():
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

            (_, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores), _ = \
                model(input_ids, input_mask, text_len, speaker_ids, genre,
                      is_training, gold_starts, gold_ends,
                      cluster_ids, sentence_map)

            predicted_antecedents = model.get_predicted_antecedents(top_antecedents.cpu(),
                                                                    top_antecedent_scores.cpu())
            # 预测实体索引
            example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                            predicted_antecedents)
            # 索引——>文字
            example_sentence = utils.flatten(example["sentences"])
            predicted_list = []
            for same_entity in example["predicted_clusters"]:
                same_entity_list = []
                num_same_entity = len(same_entity)
                for index in range(num_same_entity):
                    entity_name = ''.join(example_sentence[same_entity[index][0]: same_entity[index][1] + 1])
                    same_entity_list.append(entity_name)
                predicted_list.append(same_entity_list)
                same_entity_list = []  # 清空list
            example["predicted_idx2entity"] = predicted_list

            example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
            example['head_scores'] = []

            output_file.write(json.dumps(example, ensure_ascii=False))
            output_file.write("\n")

if __name__ == "__main__":

    os.environ["data_dir"] = "./data"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    run_experiment = "bert_base_chinese"
    config = utils.read_config(run_experiment, "experiments.conf")
    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(41)
    else:
        torch.manual_seed(41)

    # 训练阶段
    if config["do_train"]:
        train_coref(config)

    # 验证阶段
    if config["do_eval"]:
        try:
            eval_coref(config)
        except AttributeError as exception:
            print("Found too many repeated mentions (> 10) in the response, so refusing to score")

    # 测试阶段
    if config["do_test"]:
        test_coref(config)

    # 单句样本测试
    if config["do_one_example_test"]:
        input_text = "我的偶像是姚明，他喜欢打篮球，他的老婆叫叶莉。"
        online_test_coref(config, input_text)
