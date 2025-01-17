"""
DISAE finetunig model training
"""


# -------------
import os
import argparse
import logging
import random

# import time
from datetime import datetime
import torch

# import torch.optim as optim
# import numpy as np
from rdkit.Chem import MolFromSmiles

# -------------    from Huggingface, downloaded in Jan 2020
from transformers import BertTokenizer

# from transformers import AlbertTokenizer
from transformers import AlbertConfig

# -------------
from .models import MolecularGraphCoupler
from .trainer import Trainer
from .utils import (
    load_edges_from_file,
    # load_ikey2smiles,
    save_json,
    load_json,
    str2bool,
)
from .evaluator import Evaluator


def set_hyperparameters():
    parser = argparse.ArgumentParser("Train DISAE based classifier")

    # args for ALBERT model
    parser.add_argument(
        "--protein_embedding_type",
        type=str,
        default="albert",
        help="albert, lstm are available options",
    )
    parser.add_argument(
        "--frozen_list",
        type=str,
        default="8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23",
        help="enable module based frozen ALBERT",
    )
    parser.add_argument(
        "--prot_feature_size",
        type=int,
        default=256,
        help="protein representation dimension",
    )
    parser.add_argument(
        "--prot_max_seq_len",
        type=int,
        default=256,
        help="maximum length of a protein sequence including special tokens",
    )
    parser.add_argument(
        "--prot_dropout",
        type=float,
        default=0.1,
        help="Dropout prob for protein representation",
    )
    # args for LSTM protein Embedding
    parser.add_argument(
        "--lstm_embedding_size",
        type=int,
        default=128,
        help="protein representation dimension for LSTM",
    )
    parser.add_argument(
        "--lstm_num_layers", type=int, default=3, help="num LSTM layers"
    )
    parser.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=64,
        help="protein representation dimension for LSTM",
    )
    parser.add_argument(
        "--lstm_out_size",
        type=int,
        default=128,
        help="protein representation dimension for LSTM",
    )
    parser.add_argument(
        "--lstm_input_dropout",
        type=float,
        default=0.2,
        help="Dropout prob for protein representation",
    )
    parser.add_argument(
        "--lstm_output_dropout",
        type=float,
        default=0.3,
        help="Dropout prob for protein representation",
    )
    # parameters for the chemical
    parser.add_argument(
        "--chem_dropout",
        type=float,
        default=0.1,
        help="Dropout prob for chemical fingerprint",
    )
    parser.add_argument(
        "--chem_conv_layer_sizes",
        type=list,
        default=[20, 20, 20, 20],
        help="Conv layers for chemicals",
    )
    parser.add_argument(
        "--chem_feature_size",
        type=int,
        default=128,
        help="chemical fingerprint dimension",
    )
    parser.add_argument(
        "--chem_degrees",
        type=list,
        default=[0, 1, 2, 3, 4, 5],
        help="Atomic connectivity degrees for chemical molecules",
    )
    # args for Attentive Pooling
    parser.add_argument(
        "--ap_dropout",
        type=float,
        default=0.1,
        help="Dropout prob for chem&prot during attentive pooling",
    )
    parser.add_argument(
        "--ap_feature_size",
        type=int,
        default=64,
        help="attentive pooling feature dimension",
    )
    # args for model training and optimization
    parser.add_argument(
        "--datapath",
        default="Data/HMDB/activity",
        help="Path to the train/dev dataset.",
    )
    parser.add_argument(
        "--prediction_mode",
        default="binary",
        type=str,
        help="set to continuous and provide pretrained checkpoint",
    )
    parser.add_argument(
        "--pretrained_checkpoint_dir",
        default="temp/",
        help="Directory where pretrained checkpoints are saved. "
        "ignored if --from_pretrained_checkpoint is false",
    )
    parser.add_argument("--random_seed", default=705, help="Random seed.")
    parser.add_argument(
        "--epoch", default=3, type=int, help="Number of training epoches (default 50)"
    )
    parser.add_argument(
        "--batch", default=64, type=int, help="Batch size. (default 64)"
    )
    parser.add_argument(
        "--max_eval_steps",
        default=1000,
        type=int,
        help="Max evaluation steps. (nsamples=batch*steps)",
    )
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosineannealing",
        help="scheduler to adjust learning rate [cyclic or cosineannealing]",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument(
        "--l2", type=float, default=1e-4, help="L2 regularization weight"
    )
    parser.add_argument(
        "--num_threads", default=8, type=int, help="Number of threads for torch"
    )
    parser.add_argument(
        "--log", default="INFO", help="Logging level. Set to DEBUG for more details."
    )
    parser.add_argument(
        "--no_cuda",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Disables CUDA training.",
    )
    # to load data
    parser.add_argument("--trainset", type=str, default="train.tsv")
    parser.add_argument("--devset", type=str, default="dev.tsv")
    parser.add_argument("--testset", type=str, default="test.tsv")

    opt = parser.parse_args()
    opt.frozen_list = [int(f) for f in opt.frozen_list.split(",")]
    return opt


def set_folders(opt, save_folder="experiment_logs/"):
    # ---------- set folders ----------
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    print("timestamp: ", timestamp)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    checkpoint_dir = "{}/exp{}/".format(save_folder, timestamp)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    seed = opt.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return checkpoint_dir


def set_logging(opt, format="%(asctime)-15s %(message)s"):
    logging.basicConfig(format=format, level=getattr(logging, opt.log.upper()))
    logging.info(opt)


def set_up_pretrained_albert(opt, data_path, vocab, config, checkpoint, vocab_size):
    opt.albertdatapath = data_path
    opt.albertvocab = os.path.join(opt.albertdatapath, vocab)
    opt.albertconfig = os.path.join(opt.albertdatapath, config)
    opt.albert_pretrained_checkpoint = os.path.join(opt.albertdatapath, checkpoint)
    opt.lstm_vocab_size = vocab_size
    return AlbertConfig.from_pretrained(opt.albertconfig)


def set_up_data(opt):
    opt.traindata = os.path.join(
        opt.datapath, opt.trainset
    )  # train data for binary labels
    opt.devdata = os.path.join(opt.datapath, opt.devset)  # dev data for binary labels
    opt.testdata = os.path.join(opt.datapath, opt.testset)  # dev data for binary labels

    logging.info("Loading protein representations...")

    hmdb2triplets = load_json("Data/HMDB/protein/hmdbid2triplets.json")

    for uni in hmdb2triplets.keys():
        triplets = hmdb2triplets[uni].strip().split(" ")
        triplets.pop(0)
        triplets.pop(-1)
        hmdb2triplets[uni] = " ".join(triplets)

    logging.info(
        """
        Protein representations successfully loaded.
        Loading protein-ligand interactions.
        """
    )

    train_edges, train_ikeys, train_uniprots = load_edges_from_file(
        opt.traindata, sep="\t", header=False
    )

    dev_edges, dev_ikeys, dev_uniprots = load_edges_from_file(
        opt.devdata, sep="\t", header=False
    )

    test_edges, test_ikeys, test_uniprots = load_edges_from_file(
        opt.testdata, sep="\t", header=False
    )
    logging.info("Protein-ligand interactions successfully loaded.")
    torch.set_num_threads(opt.num_threads)

    file_path = "Data/HMDB/integrated/chemicals"
    hmdbacc2smiles = {}
    with open(os.path.join(file_path, "integrated_chemicals.tsv"), "r") as fin:
        for line in fin:
            line = line.strip().split("\t")
            hmdbacc = line[1]
            smi = line[2]
            hmdbacc2smiles[hmdbacc] = smi

    hmdbacc2mol = {}
    hmdbaccs = list(set(train_ikeys + dev_ikeys + test_ikeys))
    for hmdbacc in hmdbaccs:
        try:
            mol = MolFromSmiles(hmdbacc2smiles[hmdbacc])
            hmdbacc2mol[hmdbacc] = mol
        except KeyError:
            continue
    return (
        train_edges,
        train_ikeys,
        train_uniprots,
        dev_edges,
        dev_ikeys,
        dev_uniprots,
        test_edges,
        test_ikeys,
        test_uniprots,
        hmdb2triplets,
        hmdbacc2smiles,
        hmdbacc2mol,
    )


def set_up_finetuning_models(opt, albertconfig, checkpoint_dir):
    berttokenizer = BertTokenizer.from_pretrained(opt.albertvocab)

    model = MolecularGraphCoupler(
        protein_embedding_type=opt.protein_embedding_type,  # could be albert, LSTM,
        prediction_mode=opt.prediction_mode,
        # protein features - albert
        albertconfig=albertconfig,
        tokenizer=berttokenizer,
        ckpt_path=opt.albert_pretrained_checkpoint,
        frozen_list=opt.frozen_list,
        # protein features - LSTM
        lstm_vocab_size=opt.lstm_vocab_size,
        lstm_embedding_size=opt.lstm_embedding_size,
        lstm_hidden_size=opt.lstm_hidden_size,
        lstm_num_layers=opt.lstm_num_layers,
        lstm_out_size=opt.lstm_out_size,
        lstm_input_dropout_p=opt.lstm_input_dropout,
        lstm_output_dropout_p=opt.lstm_output_dropout,
        # chemical features
        conv_layer_sizes=opt.chem_conv_layer_sizes,
        output_size=opt.chem_feature_size,
        degrees=opt.chem_degrees,
        # attentive pooler features
        ap_hidden_size=opt.ap_feature_size,
        ap_dropout=opt.ap_dropout,
    )
    config_path = os.patn.join(checkpoint_dir, "config.json")
    save_json(vars(opt), config_path)
    logging.info("model configurations saved to {}".format(config_path))
    if torch.cuda.is_available():
        logging.info("Moving model to GPU ...")
        model = model.cuda()
        logging.debug("Done")
    else:
        model = model.cpu()
        logging.debug("Running on CPU...")
    return model, berttokenizer


if __name__ == "__main__":
    opt = set_hyperparameters()
    checkpoint_dir = set_folders(opt)
    set_logging(opt)
    albertconfig = set_up_pretrained_albert(
        opt,
        data_path="Data/DISAE_data/albertdata/",
        vocab="vocab/pfam_vocab_triplets.txt",
        config="albertconfig/albert_config_tiny_google.json",
        checkpoint="pretrained_whole_pfam/model.ckpt-1500000",
        vocab_size=19688,
    )

    # ---------- set up data ----------
    (
        edges,
        train_ikeys,
        train_uniprots,
        dev_edges,
        dev_ikeys,
        dev_uniprots,
        test_edges,
        test_ikeys,
        test_uniprots,
        hmdb2triplets,
        hmdbacc2smiles,
        hmdbacc2mol,
    ) = set_up_data(opt)

    # ---------- set up fine-tuning models ----------
    model, berttokenizer = set_up_finetuning_models(opt, albertconfig, checkpoint_dir)

    # -------------------------------------------
    #      set up trainer and evaluator
    # -------------------------------------------

    trainer = Trainer(
        model=model,
        berttokenizer=berttokenizer,
        epoch=opt.epoch,
        batch_size=opt.batch,
        ckpt_dir=checkpoint_dir,
        optimizer=opt.optimizer,
        l2=opt.l2,
        lr=opt.lr,
        scheduler=opt.scheduler,
        chemid2smiles=hmdbacc2smiles,
        chemid2mol=hmdbacc2mol,
        uniprot2triplets=hmdb2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
    )

    train_evaluator = Evaluator(
        chemid2smiles=hmdbacc2smiles,
        chemid2mol=hmdbacc2mol,
        berttokenizer=berttokenizer,
        protid2triplets=hmdb2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
        datatype="train",
        max_steps=opt.max_eval_steps,
        batch=opt.batch,
        shuffle=True,
    )

    dev_evaluator = Evaluator(
        chemid2smiles=hmdbacc2smiles,
        chemid2mol=hmdbacc2mol,
        berttokenizer=berttokenizer,
        protid2triplets=hmdb2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
        datatype="dev",
        max_steps=opt.max_eval_steps,
        batch=opt.batch,
        shuffle=False,
    )

    test_evaluator = Evaluator(
        chemid2smiles=hmdbacc2smiles,
        chemid2mol=hmdbacc2mol,
        berttokenizer=berttokenizer,
        protid2triplets=hmdb2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
        datatype="dev",
        max_steps=opt.max_eval_steps,
        batch=opt.batch,
        shuffle=False,
    )
    logging.debug("Train and Dev evaluators initialized.\nStart training...")

    # -------------------------------------------
    #      training and evaluating
    # -------------------------------------------
    (
        record_dict,
        loss_train,
        f1_train,
        auc_train,
        aupr_train,
        f1_dev,
        auc_dev,
        aupr_dev,
    ) = trainer.train(
        edges,
        train_evaluator,
        dev_edges,
        dev_evaluator,
        test_edges,
        test_evaluator,
        checkpoint_dir,
    )

    record_path = checkpoint_dir + "training_record.json"
    save_json(record_dict, record_path)
    logging.info("Training record saved to {}".format(record_path))

    print("Training record saved to {}".format(record_path))
