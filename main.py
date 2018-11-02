#!/anaconda3/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - songheqi <songheqi1996@gmail.com>

import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import SpecModel
from utils import get_sentence, get_transform, preprocess_data, BatchManager, load_wordvec
from conlleval import return_report

## Session configuration\
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0


## hyperparameters
parser = argparse.ArgumentParser(description='Transfer Learning on BiLSTM-CRF for Chinese NER task')
parser.add_argument('--embedding_size', type=int, default=100, help='char embedding_dim')
parser.add_argument('--hidden_size', type=int, default=150, help='dim of lstm hidden state')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--PROJ', type=str, default='Linear', help='use domain masks or not')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--project_size', type=int, default=150, help='dim of project hidden state')
parser.add_argument('--batch_size', type=int, default=20, help='#sample of each minibatch')
parser.add_argument('--mode', type=str, default='train', help='mode of want')
parser.add_argument('--train_data', type=str, default='data/train', help='normal train data')
parser.add_argument('--test_data', type=str, default='data/test', help='normal test data')
parser.add_argument('--transfer_train_data', type=str, default='data/transfer_train', help='transfer train data')
parser.add_argument('--transfer_test_data', type=str, default='data/transfer_test', help='transfer train data')
parser.add_argument('--model_path', type=str, default='ckpt', help='path to save model')
parser.add_argument('--map_path', type=str, default='data/maps.pkl', help='path to save maps')
parser.add_argument('--wiki_path', type=str, default='data/wiki_100.utf8', help='wiki chinese embeddings')
# parser.add_argument('--transfer_map_path', type=str, default='data/transfer_maps.pkl', help='path to save maps of transfer')
parser.add_argument('--tag2label_path', type=str, default='data/tag2label.json', help='config tag2label')
parser.add_argument('--transfer_tag2label_path', type=str, default='data/transfer_tag2label.json', help='config transfer tag2label')
# parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()


def get_train_data():
    normal_train, normal_test = get_sentence(args.train_data, args.test_data)
    transfer_train, transfer_test = get_sentence(args.transfer_train_data, args.transfer_test_data)
    char2id, id2char, tag2id, id2tag, transfer_tag2id, transfer_id2tag = get_transform(normal_train + transfer_train, 
                                                                                       args.map_path,
                                                                                       args.tag2label_path,
                                                                                       args.transfer_tag2label_path)
    train_data = preprocess_data(normal_train, char2id, tag2id)
    train_manager = BatchManager(train_data, args.batch_size)
    test_data = preprocess_data(normal_test, char2id, tag2id)
    test_manager = BatchManager(test_data, args.batch_size)
    transfer_train_data = preprocess_data(transfer_train, char2id, transfer_tag2id)
    transfer_train_manager = BatchManager(transfer_train_data, args.batch_size)
    transfer_test_data = preprocess_data(transfer_test, char2id, transfer_tag2id)
    transfer_test_manager = BatchManager(transfer_test_data, args.batch_size)

    return train_manager, test_manager, transfer_train_manager, transfer_test_manager, id2char, id2tag, transfer_id2tag
    


def train(max_epoch=40):
    train_manager, test_manager, transfer_train_manager, transfer_test_manager, id2char, id2tag, transfer_id2tag = get_train_data()
    with tf.Session() as sess:
        normal_model = SpecModel(args=args,
                                 num_tags=len(id2tag),
                                 vocab_size=len(id2char),
                                 name='normal')
        transfer_model = SpecModel(args=args,
                                   num_tags=len(transfer_id2tag),
                                   vocab_size=len(id2char),
                                   name='transfer')
        normal_model.build()
        transfer_model.build()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(args.model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with random parameters")
            sess.run(tf.global_variables_initializer())
            embeddings = sess.run(normal_model.get_embeddings().read_value())
            embeddings = load_wordvec(args.wiki_path, id2char, args.embedding_size, embeddings)
            sess.run(normal_model.get_embeddings().assign(embeddings))
        print("========== Start training ==========")
        for i in range(max_epoch):
            loss = []
            transfer_loss = []
            for batch, transfer_batch in zip(train_manager.iter_batch(), transfer_train_manager.iter_batch()):
                step, batch_loss = normal_model.run_one_step(sess, True, batch)
                loss.append(batch_loss)
                if step % 1000 == 0:
                    print("Step: %d Loss: %f" % (step, batch_loss))
                transfer_step, transfer_batch_loss = transfer_model.run_one_step(sess, True, transfer_batch)
                transfer_loss.append(transfer_batch_loss)
                if transfer_step % 1000 == 0:
                    print("Step: %d Transfer Loss: %f" % (transfer_step, transfer_batch_loss))
            print("Epoch: {} Loss: {:>9.6f}".format(i, np.mean(loss)))
            results = normal_model.evaluate(sess, test_manager, id2tag)
            for line in test_ner(results, "data/test_result"):
                print(line)
            print("Epoch: {} Transfer Loss: {:>9.6f}".format(i, np.mean(transfer_loss)))
            results = transfer_model.evaluate(sess, transfer_test_manager, transfer_id2tag)
            for line in test_ner(results, "data/transfer_test_result"):
                print(line)
            ckpt_file = os.path.join(args.model_path, str(i) + "ner.ckpt")
            saver.save(sess, ckpt_file)
        print("========== Finish training ==========")


def test_ner(results, path):
    output_file = os.path.join(path)
    with open(output_file, "w", encoding='utf-8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


if __name__ == "__main__":
    if args.mode == 'train':
        train(200)