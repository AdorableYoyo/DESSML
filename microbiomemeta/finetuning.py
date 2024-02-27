import os
import argparse
import logging
import random
import json
import pickle as pk
from datetime import datetime
import wandb

from rdkit.Chem import MolFromSmiles
import torch
from torch_geometric.data import Data

# -------------    from Huggingface, downloaded in Jan 2020
from transformers import BertTokenizer
from transformers import AlbertConfig

# from transformers import AlbertTokenizer

# -------------
import sys
#sys.path.append('DESSML/')
#sys.path.append('DESSML/DISAE/')

from DISAE.models import MolecularGraphCoupler_MC
from DISAE.trainer import Trainer
from DISAE.trainer_meta import Trainer_Meta
from DISAE.utils import (
    load_edges_from_file,
    # load_ikey2smiles,
    save_json,
    load_json,
    str2bool,
)
from DISAE.evaluator import Evaluator


# the module logger
md_logger = logging.getLogger(__name__)


def set_hyperparameters():
    """ Setup hyperparameters

    Return:
        opt (Namespace): command line arguments.
    """
    parser = argparse.ArgumentParser("Train DISAE based classifier")

    # args for ALBERT model
    parser.add_argument(
        "--protein_embedding_type",
        type=str,
        default="albert",
        help="albert, lstm are available options",
    )
    parser.add_argument(
        "--frozen_list", type=str, default=[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], help="enable module based frozen ALBERT",
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
        type=lambda s: list(map(int, s.split(","))),
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
        type=lambda s: list(map(int, s.split(","))),
        default=[0, 1, 2, 3, 4, 5],
        help="Atomic connectivity degrees for chemical molecules",
    )
    parser.add_argument(
        "--albert_checkpoint",
        type=str,
        default="/raid/home/yoyowu/DESSML/Data/DISAE_data/albertdata/pretrained_whole_pfam/model.ckpt-1500000",
        help="Checkpoint path for pretrained albert.",
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="gin",
        help="Ligand embedding type. 'nf' (neuralfingerprint) or 'gin'.",
    )
    parser.add_argument(
        "--gin_checkpoint",
        type=str,
        default=None,
        help="Pre-trained weights for GIN model.",
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
        "--train_datapath",
        default="/raid/home/yoyowu/DESSML/Data/Combined/activities/combined_all/train_100.tsv",
        help="Path to the train dataset.",
    )
    parser.add_argument(
        "--dev_datapath",
        default="/raid/home/yoyowu/DESSML/Data/Combined/activities/combined_all/dev_100.tsv",
        help="Path to the dev dataset.",
    )
    parser.add_argument(
        "--test_datapath",
        default="/raid/home/yoyowu/DESSML/Data/TestingSetFromPaper/activities_nolipids.txt",
        help="Path to the test dataset.",
    )
    parser.add_argument(
        "--prot2trp_path",
        type=str,
        default="/raid/home/yoyowu/DESSML/Data/Combined/proteins/triplets_in_my_data_set.pk",
        help="path of the protein id to triplets mapping json file.",
    )
    parser.add_argument(
        "--chm2smiles_path",
        type=str,
        default="/raid/home/yoyowu/DESSML/Data/Combined/chemicals/combined_compounds.pk",
        help="path of the chemical id to SMILES mapping tsv file.",
    )
    parser.add_argument(
        "--prediction_mode",
        default="binary",
        type=str,
        help="set to continuous and provide pretrained checkpoint",
    )
    parser.add_argument(
        "--pretrained_checkpoint_dir",
        default=None,
        help="Directory where pretrained checkpoints are saved. "
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
        default=1,
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
    parser.add_argument(
        "--freezing_epochs",
        type=int,
        default=float("inf"),
        help="Number of epochs to freeze NLP model.",
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument(
        "--l2", type=float, default=1e-4, help="L2 regularization weight"
    )
    parser.add_argument(
        "--num_threads", default=8, type=int, help="Number of threads for torch"
    )
    parser.add_argument(
        "--log",
        default="INFO",
        type=str,
        help="Logging level. Set to DEBUG for more details.",
    )
    parser.add_argument(
        "--no_cuda",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Disables CUDA training.",
    )
    parser.add_argument("--meta_training", type=bool, default=False, help="Use meta training setup. To implement MAML mentioned in the ablation study")
    parser.add_argument("--update_step", default=5, type=int, help="Number of meta update steps")
    parser.add_argument("--global_meta", default=30000, type=int, help="Use global meta training")
    parser.add_argument("--global_eval_at", default=300, type=int, help="Global evaluation at for meta learning ")
    parser.add_argument("--task_num", default=5, type=int, help="Number of tasks for meta learning")
    parser.add_argument("--test_run", action="store_true", default=False,help="Use testing setup.")
    parser.add_argument("--exp_id", default="debug_meta_01", help="the run name")
    opt = parser.parse_args()
    if isinstance(opt.frozen_list, str):
        opt.frozen_list = [int(f) for f in opt.frozen_list.split(",")]
    return opt


def set_folders(opt, save_folder="experiment_logs/"):
    """ Set up folders to save the fine-tuning results.

    Args:
        opt (Namespace): arguments.
        save_folder (str): root path of the saving directory. A sub-directory will be
            created under the root directory to save the outputs. The name of the
            sub-directory is exp{current_time}.

    Return:
        checkpoint_dir (str): path to the sub-directory.
    """
    # ---------- set folders ----------
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    md_logger.info(f"timestamp: {timestamp}")
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


def set_logging(opt, format="%(asctime)-15s %(name)s [%(levelname)s]: %(message)s"):
    """ Set up logging configurations.

    Args:
        opt (Namespace): arguments.
        format (str): logging format. Default: "%(asctime)-15s %(name)s [%(levelname)s]:
            %(message)s"
    """

    logging.basicConfig(
        format=format,
        level=getattr(logging, opt.log.upper()),
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    md_logger.info(opt)


def set_up_pretrained_albert(opt, data_path, vocab, config, checkpoint, vocab_size):
    """ Set up the pretrained ALBERT model. Add ALBERT's vocabulary path, checkpoint
    path, and vocabulary size to the argument Namespace.

    Args:
        opt (Namespace): arguments.
        data_path (str): path to the albert data root directory.
        vocab (str): vocabulary filename.
        config (str): config filename.
        checkpoint (str): checkpoint path for pretrained albert.
        vocab_size (int): size of the vocabulary.

    Return:
        AlbertConfig: the configuration for the ALBERT model based on the config file.
    """
    opt.albertdatapath = data_path
    opt.albertvocab = os.path.join(opt.albertdatapath, vocab)
    opt.albertconfig = os.path.join(opt.albertdatapath, config)
    opt.albert_pretrained_checkpoint = checkpoint
    opt.lstm_vocab_size = vocab_size
    return AlbertConfig.from_pretrained(opt.albertconfig)


def set_up_data(opt):
    """ Set up fine-tuning data.

    Args:
        opt (Namespace): arguments.

    Returns:
        train_edges (dict): keys are compound id + protein id, values are labels for
          keys.
        train_chem_ids (list): id list of chemical compounds.
        train_prot_ids (list): id list of proteins.
        dev_edges (dict): the same as train_edges.
        dev_chem_ids (list): the same as train_chem_ids.
        dev_prot_ids (list): the same as train_prot_ids.
        test_edges (dict): the same as train_edges.
        test_chem_ids (list): the same as train_chem_ids.
        test_prot_ids (list): the same as train_prot_ids.
        proteinid2triplets (dict): the map from protein ids to their triplets.
        chemicalid2smiles (dict): the map from chemical compounds ids to their SMILES
          strings.
        chemicalid2mol (dict): the map from chemical compounds ids to their RDKit
          Molecule objects.
    """

    md_logger.info("Loading protein representations...")
    if opt.test_run:
        md_logger.info("Using test run setup.")

    if opt.prot2trp_path.endswith(".json"):
        proteinid2triplets = load_json(opt.prot2trp_path)
    elif opt.prot2trp_path.endswith(".pk"):
        with open(opt.prot2trp_path, "rb") as f:
            proteinid2triplets = pk.load(f)

    # for uni in proteinid2triplets.keys():
    #     triplets = proteinid2triplets[uni].strip().split(" ")[1:-1]
    #     proteinid2triplets[uni] = " ".join(triplets)

    md_logger.info(
        """
        Protein representations successfully loaded.
        Loading protein-ligand interactions.
        """
    )

    if opt.test_run:
        train_edges, train_chem_ids, train_prot_ids = load_edges_from_file(
            "test/Combined_activity/train.tsv", sep="\t", header=False
        )
        dev_edges, dev_chem_ids, dev_prot_ids = load_edges_from_file(
            "test/Combined_activity/dev.tsv", sep="\t", header=False
        )
        test_edges, test_chem_ids, test_prot_ids = load_edges_from_file(
            "test/Combined_activity/test.tsv", sep="\t", header=False
        )
    else:
        train_edges, train_chem_ids, train_prot_ids = load_edges_from_file(
            opt.train_datapath, sep="\t", header=False
        )
        dev_edges, dev_chem_ids, dev_prot_ids = load_edges_from_file(
            opt.dev_datapath, sep="\t", header=False
        )
        test_edges, test_chem_ids, test_prot_ids = load_edges_from_file(
            opt.test_datapath, sep="\t", header=False
        )

    md_logger.info("Protein-ligand interactions successfully loaded.")

    md_logger.info("Loading compounds mapping...")
    if opt.chm2smiles_path.endswith(".tsv"):
        chemicalid2smiles = {}
        with open(opt.chm2smiles_path, "r") as fin:
            for line in fin:
                line = line.strip().split("\t")
                chemid = line[1]
                smi = line[2]
                chemicalid2smiles[chemid] = smi
    elif opt.chm2smiles_path.endswith(".json"):
        with open(opt.chm2smiles_path) as f:
            chemicalid2smiles = json.load(f)
    elif opt.chm2smiles_path.endswith(".pk"):
        with open(opt.chm2smiles_path, "rb") as f:
            chemicalid2smiles = pk.load(f)

    if isinstance(next(iter(chemicalid2smiles.values())), Data):
        chemicalid2mol = chemicalid2smiles
    else:
        chemicalid2mol = {}
        for chemid, smiles in chemicalid2smiles.items():
            mol = MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(
                    f"Failed to convert {chemid}: {smiles} to RDKit Mol object."
                )
            chemicalid2mol[chemid] = mol
    md_logger.info("Chemical compounds mapping successfully loaded.")

    return (
        train_edges,
        train_chem_ids,
        train_prot_ids,
        dev_edges,
        dev_chem_ids,
        dev_prot_ids,
        test_edges,
        test_chem_ids,
        test_prot_ids,
        proteinid2triplets,
        chemicalid2smiles,
        chemicalid2mol,
    )


def set_up_finetuning_models(opt, albertconfig, gin_config, checkpoint_dir,ckpt_path=None):
    """ Set up the model.

    Args:
        opt (NameSpace): arguments.
        albertconfig (AlbertConfig): the configuration object for the transformer model.
        gin_config (dict): the configuration for the gin model.
        checkpoint_dir (str): path to the directory where pre-trained weights are saved.

    Returns:
        model: the MolecularGraphCoupler_MC model.
        berttokenizer: tokenizer for the BERT model.
    """
    berttokenizer = BertTokenizer.from_pretrained(opt.albertvocab)

    model = MolecularGraphCoupler_MC(
        nclass=2,
        protein_embedding_type=opt.protein_embedding_type,  # could be albert, LSTM,
        gnn_type=opt.gnn_type,  # could be "nf", "gin"
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
        gin_config=gin_config,
    )
    config_path = os.path.join(checkpoint_dir, "config.json")
    save_json(vars(opt), config_path)
    md_logger.info("model configurations saved to {}".format(config_path))
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
        md_logger.info(f"Model weights loaded from {ckpt_path}")
    if torch.cuda.is_available():
        md_logger.info("Moving model to GPU ...")
        model = model.cuda()
    else:
        model = model.cpu()
        md_logger.info("Running on CPU...")
    return model, berttokenizer


if __name__ == "__main__":
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    opt = set_hyperparameters()
    set_logging(opt)
    checkpoint_dir = set_folders(opt)
    albertconfig = set_up_pretrained_albert(
        opt,
        data_path="/raid/home/yoyowu/DESSML/Data/DISAE_data/albertdata/",
        vocab="vocab/pfam_vocab_triplets.txt",
        config="albertconfig/albert_config_tiny_google.json",
        checkpoint=opt.albert_checkpoint,
        vocab_size=19688,
    )
    wandb.init(project="microbio_meta",config=opt, name = opt.exp_id)
    wandb.define_metric("dev AUPRC",summary="max")
    wandb.define_metric("dev AUROC",summary="max")
    wandb.define_metric("test AUPRC",summary="max")
    wandb.define_metric("test AUROC",summary="max")

    if opt.gnn_type == "gin":
        gin_config = {
            "num_layer": 5,
            "emb_dim": 300,
            "JK": "last",
            "drop_ratio": 0.2,
            "checkpoint": opt.gin_checkpoint,
        }
    else:
        gin_config = None

    torch.set_num_threads(opt.num_threads)

    #---------- set up data ----------
    (
        edges,
        train_chem_ids,
        train_prot_ids,
        dev_edges,
        dev_chem_ids,
        dev_prot_ids,
        test_edges,
        test_chem_ids,
        test_prot_ids,
        proteinid2triplets,
        chemicalid2smiles,
        chemicalid2mol,
    ) = set_up_data(opt)

    # ---------- set up fine-tuning models ----------
    md_logger.info("Setting up model...")
    model, berttokenizer = set_up_finetuning_models(
        opt, albertconfig, gin_config, checkpoint_dir, opt.pretrained_checkpoint_dir
    )
    md_logger.info("Model successfully set up.")

    # -------------------------------------------
    #      set up trainer and evaluator
    # -------------------------------------------
    if opt.meta_training:
        print("Using meta training setup.")
        trainer = Trainer_Meta(
            model=model,
            berttokenizer=berttokenizer,
            batch_size=opt.batch,
            ckpt_dir=checkpoint_dir,
            optimizer=opt.optimizer,
            l2=opt.l2,
            lr=opt.lr,
            scheduler=opt.scheduler,
            chemid2smiles=chemicalid2smiles,
            chemid2mol=chemicalid2mol,
            uniprot2triplets=proteinid2triplets,
            prediction_mode=opt.prediction_mode,
            protein_embedding_type=opt.protein_embedding_type,
            freezing_epochs=opt.freezing_epochs,
            update_step = opt.update_step,
            global_meta = opt.global_meta,
            global_eval_at = opt.global_eval_at,
            task_num = opt.task_num

        )
    else:
        print("Using standard training setup.")
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
            chemid2smiles=chemicalid2smiles,
            chemid2mol=chemicalid2mol,
            uniprot2triplets=proteinid2triplets,
            prediction_mode=opt.prediction_mode,
            protein_embedding_type=opt.protein_embedding_type,
            freezing_epochs=opt.freezing_epochs,
        )

    train_evaluator = Evaluator(
        chemid2smiles=chemicalid2smiles,
        chemid2mol=chemicalid2mol,
        berttokenizer=berttokenizer,
        uniprot2triplets=proteinid2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
        datatype="train",
        max_steps=opt.max_eval_steps,
        batch=opt.batch,
        shuffle=True,
    )

    dev_evaluator = Evaluator(
        chemid2smiles=chemicalid2smiles,
        chemid2mol=chemicalid2mol,
        berttokenizer=berttokenizer,
        uniprot2triplets=proteinid2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
        datatype="dev",
        max_steps=opt.max_eval_steps,
        batch=opt.batch,
        shuffle=False,
    )

    test_evaluator = Evaluator(
        chemid2smiles=chemicalid2smiles,
        chemid2mol=chemicalid2mol,
        berttokenizer=berttokenizer,
        uniprot2triplets=proteinid2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
        datatype="test",
        max_steps=opt.max_eval_steps,
        batch=opt.batch,
        shuffle=False,
    )
    md_logger.info("Train and Dev evaluators initialized.\nStart training...")

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
    ) =trainer.train(
        edges,
        train_evaluator,
        dev_edges,
        dev_evaluator,
        test_edges,
        test_evaluator,
        checkpoint_dir,
    )

    record_path = os.path.join(checkpoint_dir, "training_record.json")
    save_json(record_dict, record_path)
    md_logger.info("Training record saved to {}".format(record_path))
    print("Training record saved to {}".format(record_path))
