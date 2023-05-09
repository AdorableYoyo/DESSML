import torch.nn as nn
import torch
import torch.nn.functional as F

from .fingerprint.features import num_atom_features, num_bond_features
from .fingerprint.models import NeuralFingerprint
import logging
from transformers import BertTokenizer

from transformers import AlbertForMaskedLM
from transformers import AlbertConfig
from transformers import load_tf_weights_in_albert
from .resnet import ResnetEncoderModel
from microbiomemeta.models import GNN_graphpred


class EmbeddingRNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        num_layers,
        out_size,
        input_dropout_p=0.2,
        rnn_dropout_p=0.3,
    ):
        super(EmbeddingRNNEncoder, self).__init__()
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")

        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.embed = nn.Embedding(vocab_size, embedding_size)
        # dropout on RNN is not effective when it has only l1 layer
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=rnn_dropout_p,
        )

    def cuda(self, device_id=None):
        self.lstm.cuda(device_id=device_id)

    def forward(self, input_seq):
        # TODO: variable length rnn
        self.logger.debug("input_seq {}".format(input_seq.size()))
        embedding = self.embed(input_seq)
        self.logger.debug("embedding {}".format(embedding.size()))
        embedding = self.input_dropout(embedding)
        self.logger.debug("Apply LSTM...")
        context, _ = self.lstm(embedding)
        self.logger.debug("Done. LSTM context size {}".format(context.size()))
        return context


class MolecularGraphCoupler(nn.Module):
    # chemical embedding is by GraphDegreeCNN
    # protein embedding type can be chosen by the user
    def __init__(
        self,
        protein_embedding_type="albert",  # could be albert, LSTM,
        gnn_type="nf",  # could be gin or nf (neralfingerprint)
        prediction_mode="binary",  # could be continuous (e.g. pKi, pIC50)
        # protein features - albert
        albertconfig=None,
        tokenizer=None,
        ckpt_path=None,
        frozen_list=[22, 23],
        # protein features - LSTM
        lstm_vocab_size=26,
        lstm_embedding_size=128,
        lstm_hidden_size=64,
        lstm_num_layers=3,
        lstm_out_size=32,
        lstm_input_dropout_p=0.2,
        lstm_output_dropout_p=0.3,
        # chemical features
        conv_layer_sizes=[20, 20, 20, 20],
        output_size=128,
        degrees=[0, 1, 2, 3, 4, 5],
        # attentive pooler features
        ap_hidden_size=64,
        ap_dropout=0.1,
        gin_config=None,  # need to be provided when gnn_type is "gin",
        get_embeddings = None
    ):
        super(MolecularGraphCoupler, self).__init__()
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        self.prediction_mode = prediction_mode
        self.protein_embedding_type = protein_embedding_type
        prot_hidden_size = 256
        self.gnn_type = gnn_type
        self.get_embeddings = get_embeddings

        if self.protein_embedding_type == "albert":
            print("==================================================ALBERT")
            self.proteinEmbedding = AlbertResnet(
                albertconfig=albertconfig,
                tokenizer=tokenizer,
                ckpt_path=ckpt_path,
                frozen_list=frozen_list,
            )
            self.protein_embedding_type = "albert"
        else:
            print(
                "==========================================LSTM: VOCAB SIZE",
                lstm_vocab_size,
            )
            self.proteinEmbedding = EmbeddingRNNEncoder(
                lstm_vocab_size,
                lstm_embedding_size,
                lstm_hidden_size,
                lstm_num_layers,
                lstm_out_size,
                input_dropout_p=lstm_input_dropout_p,
                rnn_dropout_p=lstm_output_dropout_p,
            )
            self.linear_protein_pooler = EmbeddingTransform(
                210 * lstm_hidden_size, 128, lstm_hidden_size, dropout_p=ap_dropout
            )
            self.linear_ligand_pooler = EmbeddingTransform(
                210 * output_size, 128, output_size, dropout_p=ap_dropout
            )
            self.protein_embedding_type = "lstm"
            prot_hidden_size = lstm_hidden_size

        self.logger.debug(
            "Protein Embedding initialized: {}".format(protein_embedding_type.upper())
        )

        if gnn_type == "nf":
            self.ligandEmbedding = ChemicalGraphConv(
                conv_layer_sizes=conv_layer_sizes,
                output_size=output_size,
                degrees=degrees,
                num_atom_features=num_atom_features(),
                num_bond_features=num_bond_features(),
            )
        elif gnn_type == "gin":
            self.ligandEmbedding = GNN_graphpred(
                num_layer=gin_config["num_layer"],
                emb_dim=gin_config["emb_dim"],
                num_tasks=128,
                JK=gin_config["JK"],
                drop_ratio=gin_config["drop_ratio"],
            )
            if gin_config["checkpoint"] is not None:
                self.ligandEmbedding.from_pretrained(gin_config["checkpoint"])
        else:
            raise ValueError(
                f"The gnn_type {gnn_type} is illegal. Should be 'gin' or 'nf'."
            )
        self.attentive_interaction_pooler = AttentivePooling(
            chem_hidden_size=output_size, prot_hidden_size=prot_hidden_size
        )
        self.linear_interaction_pooler = EmbeddingTransform(
            output_size + prot_hidden_size, 128, ap_hidden_size, dropout_p=ap_dropout
        )
        self.binary_predictor = EmbeddingTransform(ap_hidden_size, 64, 2, dropout_p=0.2)
        self.continuous_predictor = EmbeddingTransform(
            ap_hidden_size, 64, 1, dropout_p=0.2
        )

    def forward(self, batch_input, **kwargs):
        self.logger.debug(
            "MolecularGraphCoupler: input protein {}".format(
                batch_input["protein"].size()
            )
        )

        protein_vector = self.proteinEmbedding(batch_input["protein"])
        if self.protein_embedding_type == "lstm":
            protein_vector = protein_vector
        else:
            protein_vector = protein_vector.reshape(
                batch_input["protein"].size()[0], 1, -1
            )

        self.logger.debug(
            "MolecularGraphCoupler: protein_vector {}".format(protein_vector.size())
        )
        ligand_vector = self.ligandEmbedding(batch_input["ligand"])
        if len(ligand_vector.shape) == 2:  # shape match
            ligand_vector = ligand_vector.unsqueeze(1)
        self.logger.debug(
            "MolecularGraphCoupler: ligand_vector {}".format(ligand_vector.size())
        )

        (
            (ligand_vector, ligand_score),
            (protein_vector, protein_score),
        ) = self.attentive_interaction_pooler(ligand_vector, protein_vector)
        self.logger.debug(
            "Attentive pooled ligand {}, protein {}".format(
                ligand_vector.size(), protein_vector.size()
            )
        )
        if self.protein_embedding_type == "lstm":
            protein_vector = self.linear_protein_pooler(
                protein_vector.reshape(protein_vector.size()[0], -1)
            )
            ligand_vector = self.linear_ligand_pooler(
                ligand_vector.reshape(ligand_vector.size()[0], -1)
            )
            self.logger.debug(
                "For LSTM: Additionally linear pooled ligand {}, protein {}".format(
                    ligand_vector.size(), protein_vector.size()
                )
            )
        self.logger.debug(f"protein vector size: {protein_vector.size()}")
        self.logger.debug(f"ligand vector size: {ligand_vector.size()}")
        interaction_vector = self.linear_interaction_pooler(
            torch.cat((protein_vector.squeeze(dim=1), ligand_vector.squeeze(dim=1)), 1)
        )
        self.logger.debug("interaction_vector {}".format(interaction_vector.size()))

        if self.prediction_mode.lower() == "binary":
            logits = self.binary_predictor(interaction_vector)
        else:
            logits = self.continuous_predictor(interaction_vector)
        self.logger.debug("MolecularGraphCoupler: logits {}".format(logits.size()))
        if self.get_embeddings:
            return interaction_vector
        return logits


class MolecularGraphCoupler_MC(MolecularGraphCoupler):
    """ The multi-class version of the MolecularGraphCoupler class.
    """

    def __init__(self, nclass=2, **kwargs):
        super().__init__(**kwargs)
        self.nclass = nclass
        self.binary_predictor = EmbeddingTransform(
            kwargs["ap_hidden_size"], 64, nclass, dropout_p=0.2
        )


class AlbertResnet(nn.Module):
    def __init__(
        self, albertconfig=None, tokenizer=None, ckpt_path=None, frozen_list=[22, 23]
    ):
        super(AlbertResnet, self).__init__()
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        self.config = albertconfig
        self.frozen_list = frozen_list
        if self.config is None:
            self.config = AlbertConfig.from_pretrained(
                "Data/DISAE_data/albertdata/albertconfig/albert_config_tiny_google.json"
            )
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(
                "data/albertdata/vocab/pfam_vocab_triplets.txt"
            )

        self.config.output_hidden_states = False
        model = AlbertForMaskedLM(config=self.config)
        if ckpt_path is not None:
            model = load_tf_weights_in_albert(model, self.config, ckpt_path)
            self.logger.debug("Pretrained Albert loaded from {}".format(ckpt_path))
        else:
            self.logger.debug("Albert model is not pretrained")
        self.albert = model.albert
        self._freeze_albert()

        self.resnet = ResnetEncoderModel(1)

    def _freeze_albert(self):
        ct = 0
        flag = True
        for m in self.albert.modules():
            ct += 1
            if ct in self.frozen_list:
                if flag:
                    self.logger.info(f"Frozen ALBERT modules: {self.frozen_list}")
                    flag = False
                for param in m.parameters():
                    param.requires_grad = False
            else:
                for param in m.parameters():
                    param.requires_grad = True

    def _unfreeze_albert(self):
        for m in self.albert.modules():
            for param in m.parameters():
                param.requires_grad = True

    def forward(self, batch_input, **kwargs):
        # batch_input is tensor for encoded tokens
        self.logger.debug("AlbertResnet: batch_input {}".format(batch_input.size()))
        albert_outputs = self.albert(batch_input)
        self.logger.debug(
            "AlbertResnet: albert_outputs[0] {}".format(albert_outputs[0].size())
        )
        logits = self.resnet(albert_outputs[0].unsqueeze(1))
        self.logger.debug("AlbertResnet: logits {}".format(logits.size()))
        return logits


class ChemicalGraphConv(nn.Module):
    def __init__(
        self,
        conv_layer_sizes=[20, 20, 20, 20],
        output_size=128,
        degrees=[0, 1, 2, 3, 4, 5],
        num_atom_features=num_atom_features(),
        num_bond_features=num_bond_features(),
    ):
        super(ChemicalGraphConv, self).__init__()
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        type_map = dict(batch="molecule", node="atom", edge="bond")
        self.model = NeuralFingerprint(
            num_atom_features,
            num_bond_features,
            conv_layer_sizes,
            output_size,
            type_map,
            degrees,
        )

        for param in self.model.parameters():
            param.data.uniform_(-0.08, 0.08)

    def forward(self, batch_input, **kwargs):
        batch_embedding = self.model(batch_input)
        return batch_embedding


class EmbeddingTransform(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, dropout_p=0.1):
        super(EmbeddingTransform, self).__init__()
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        self.dropout = nn.Dropout(p=dropout_p)
        self.transform = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size),
        )

    def forward(self, embedding):
        embedding = self.dropout(embedding)
        hidden = self.transform(embedding)
        return hidden


class AttentivePooling(nn.Module):
    """ Attentive pooling network according to https://arxiv.org/pdf/1602.03609.pdf """

    def __init__(self, chem_hidden_size=128, prot_hidden_size=256):
        super(AttentivePooling, self).__init__()
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        self.chem_hidden_size = chem_hidden_size
        self.prot_hidden_size = prot_hidden_size
        self.param = nn.Parameter(torch.zeros(chem_hidden_size, prot_hidden_size))

    def forward(self, first, second):
        """ Calculate attentive pooling attention weighted representation and
        attention scores for the two inputs.

        Args:
            first: output from one source with size (batch_size, length_1, hidden_size)
            second: outputs from other sources with size (batch_size, length_2,
            hidden_size)

        Returns:
            (rep_1, attn_1): attention weighted representations and attention scores
            for the first input
            (rep_2, attn_2): attention weighted representations and attention scores
            for the second input
        """
        self.logger.debug(
            "AttentivePooling first {0}, second {1}".format(first.size(), second.size())
        )
        param = self.param.expand(
            first.size(0), self.chem_hidden_size, self.prot_hidden_size
        )
        self.logger.debug("AttentivePooling params: {0}".format(param.size()))
        wm1 = torch.tanh(torch.bmm(second, param.transpose(1, 2)))
        wm2 = torch.tanh(torch.bmm(first, param))
        self.logger.debug(
            "Wm1 {}, Wm2 {} before softmax".format(wm1.size(), wm2.size())
        )
        score_m1 = F.softmax(wm1, dim=2)
        score_m2 = F.softmax(wm2, dim=2)
        self.logger.debug(
            "score_m1 {}, score_m2 {}".format(score_m1.size(), score_m2.size())
        )
        rep_first = first * score_m1
        rep_second = second * score_m2
        self.logger.debug(
            "AttentivePooling reps: {0}, {1}".format(
                rep_first.size(), rep_second.size()
            )
        )

        return ((rep_first, score_m1), (rep_second, score_m2))


class Predict(nn.Module):
    """ Prepare a similarity prediction model for each distinct pair of
    entity types.
    """

    def __init__(
        self,
        chem_hidden_size=128,
        prot_hidden_size=256,
        hidden_size=64,
        attn_dropout=0.1,
    ):
        super(Predict, self).__init__()
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        self.hidden_size = hidden_size
        self.chem_hidden_size = chem_hidden_size
        self.prot_hidden_size = prot_hidden_size
        self.add_module(
            "chemical",
            EmbeddingTransform(
                self.chem_hidden_size, hidden_size, hidden_size, dropout_p=attn_dropout
            ),
        )
        self.add_module(
            "protein",
            EmbeddingTransform(
                self.prot_hidden_size, hidden_size, hidden_size, dropout_p=attn_dropout
            ),
        )  # for temporal conv
        self.add_module(
            " ".join(("chemical", "protein")),
            AttentivePooling(
                chem_hidden_size=self.chem_hidden_size,
                prot_hidden_size=self.prot_hidden_size,
            ),
        )

        for param in self.parameters():
            param.data.uniform_(-0.08, 0.08)

    def forward(self, chem_batch, prot_batch):
        """ Calculate the 'similarity' between two inputs, where the first input
        is a matrix and the second batched matrices.

        Args:
            first: output from one source with size (length_1, hidden_size)
            second: outputs from other sources with size (batch_size, length_2,
            hidden_size)

        Returns:
            prob: a `batch_size` vector that contains the probabilities that each
            entity in the second input has association with the first input
        """
        self.logger.debug(
            "AttentivePooling inputs {}, {}".format(
                chem_batch.size(), prot_batch.size()
            )
        )
        # first, second = sorted((first, second), key=lambda x: x['type'])
        attn_model = getattr(self, "chemical protein")
        (rep_first, w_first), (rep_second, w_second) = attn_model(
            chem_batch, prot_batch
        )
        self.logger.debug(
            "rep_first {}, rep_second {}".format(rep_first.size(), rep_second.size())
        )

        rep_first = getattr(self, "chemical")(rep_first.squeeze()).unsqueeze(1)
        rep_second = getattr(self, "protein")(rep_second.squeeze()).unsqueeze(2)
        self.logger.debug(
            "Transformed representation vectors: {0}, {1}".format(
                rep_first.size(), rep_second.size()
            )
        )

        return torch.bmm(rep_first, rep_second).squeeze(), (w_first, w_second)


class PredictBinary(nn.Module):
    def __init__(
        self,
        chem_hidden_size=128,
        prot_hidden_size=256,
        hidden_size=64,
        attn_dropout=0.1,
    ):
        super(PredictBinary, self).__init__()
        self.logger = logging.getLogger(f"{__name__}.{__class__.__name__}")
        self.hidden_size = hidden_size
        self.chem_hidden_size = chem_hidden_size
        self.prot_hidden_size = prot_hidden_size
        self.add_module(
            "chemical",
            EmbeddingTransform(
                self.chem_hidden_size, hidden_size, hidden_size, dropout_p=attn_dropout
            ),
        )
        self.add_module(
            "protein",
            EmbeddingTransform(
                self.prot_hidden_size, hidden_size, hidden_size, dropout_p=attn_dropout
            ),
        )  # for temporal conv
        self.add_module(
            " ".join(("chemical", "protein")),
            AttentivePooling(
                chem_hidden_size=self.chem_hidden_size,
                prot_hidden_size=self.prot_hidden_size,
            ),
        )

        self.transform = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.BatchNorm1d(2),
        )
        for param in self.parameters():
            param.data.uniform_(-0.08, 0.08)

    def forward(self, chem_batch, prot_batch):
        """ Calculate the 'similarity' between two inputs, where the first input
        is a matrix and the second batched matrices.

        Args:
            first: output from one source with size (length_1, hidden_size)
            second: outputs from other sources with size (batch_size, length_2,
            hidden_size)

        Returns:
            prob: a `batch_size` vector that contains the probabilities that each
            entity in the second input has association with the first input
        """
        self.logger.debug(
            "AttentivePooling inputs {}, {}".format(
                chem_batch.size(), prot_batch.size()
            )
        )
        # first, second = sorted((first, second), key=lambda x: x['type'])
        attn_model = getattr(self, "chemical protein")
        (rep_first, w_first), (rep_second, w_second) = attn_model(
            chem_batch, prot_batch
        )
        self.logger.debug(
            "rep_first {}, rep_second {}".format(rep_first.size(), rep_second.size())
        )

        rep_first = getattr(self, "chemical")(rep_first.squeeze()).unsqueeze(1)
        rep_second = getattr(self, "protein")(rep_second.squeeze()).unsqueeze(2)
        self.logger.debug(
            "Transformed representation vectors: {0}, {1}".format(
                rep_first.size(), rep_second.size()
            )
        )
        output = self.transform(
            torch.cat((rep_first.squeeze(), rep_second.squeeze()), 1)
        )
        self.logger.debug(
            "attentive pooling transformation result size {}".format(output.size())
        )
        return output
