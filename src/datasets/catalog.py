from collections import namedtuple

from src.datasets.captioned_images import mscoco, vqa
from src.datasets.genomics import genomics
from src.datasets.medical_images import chexpert, xray8
from src.datasets.natural_images import aircraft, cifar, cu_birds, dtd, imagenet, traffic_sign, vgg_flower
from src.datasets.particle_physics import higgs
from src.datasets.proteins import fluorescence, pfam, scop, secondary_struct, stability
from src.datasets.satellite_images import eurosat
from src.datasets.semiconductors import wafer_map
from src.datasets.sensor import pamap2
from src.datasets.speech import audio_mnist, fluent_speech, google_speech, librispeech, voxceleb1
from src.datasets.text import glue, mc4, pawsx, wikitext

DATASET_DICT = {
    # Captioned images.
    'mscoco': mscoco.MSCOCO,
    'mismatched_caption': mscoco.MismatchedCaption,
    'vqa': vqa.VQA,

    # Medical images.
    'chexpert': chexpert.CheXpert,
    'xray8': xray8.ChestXray8,

    # Natural images.
    'aircraft': aircraft.Aircraft,
    'aircraft_small': aircraft.AircraftSmall,
    'cifar10': cifar.CIFAR10,
    'cifar10_small': cifar.CIFAR10Small,
    'cu_birds': cu_birds.CUBirds,
    'cu_birds_small': cu_birds.CUBirdsSmall,
    'dtd': dtd.DTD,
    'dtd_small': dtd.DTDSmall,
    'imagenet': imagenet.ImageNet,
    'traffic_sign': traffic_sign.TrafficSign,
    'traffic_sign_small': traffic_sign.TrafficSignSmall,
    'vgg_flower': vgg_flower.VGGFlower,
    'vgg_flower_small': vgg_flower.VGGFlowerSmall,

    # Sensor.
    'pamap2': pamap2.PAMAP2,
    'pamap2_spectrogram': pamap2.SpectrogramPAMAP2,

    # Speech.
    'audio_mnist': audio_mnist.AudioMNIST,
    'fluent_speech_action': fluent_speech.FluentSpeechAction,
    'fluent_speech_object': fluent_speech.FluentSpeechObject,
    'fluent_speech_location': fluent_speech.FluentSpeechLocation,
    'google_speech': google_speech.GoogleSpeechCommands,
    'librispeech': librispeech.LibriSpeech,
    'librispeech_transfer': librispeech.LibriSpeechTransfer,
    'voxceleb1': voxceleb1.VoxCeleb1,

    # Text.
    'cola': glue.COLA,
    'mnli_matched': glue.MNLIMatched,
    'mnli_mismatched': glue.MNLIMismatched,
    'mrpc': glue.MRPC,
    'qnli': glue.QNLI,
    'qqp': glue.QQP,
    'rte': glue.RTE,
    'sst2': glue.SST2,
    'stsb': glue.STSB,
    'wikitext103': wikitext.WikiText103,
    'wnli': glue.WNLI,
    'mc4': mc4.MC4,
    'paws_en': pawsx.PawsEN,
    'paws_fr': pawsx.PawsFR,
    'paws_es': pawsx.PawsES,
    'paws_de': pawsx.PawsDE,
    'paws_zh': pawsx.PawsZH,
    'paws_ja': pawsx.PawsJA,
    'paws_ko': pawsx.PawsKO,

    # Genomics.
    'genomics': genomics.GenomicsPretrain,
    'genomics_transfer_id': genomics.GenomicsTransferID,
    'genomics_transfer_ood': genomics.GenomicsTransferOOD,

    # Particle physics.
    'higgs': higgs.HiggsPretrain,
    'higgs_transfer': higgs.HiggsTransfer,

    # Proteins.
    'pfam': pfam.PfamPretrain,
    'pfam_transfer': pfam.PfamTransfer,
    'scop': scop.SCOP,
    'secondary_struct': secondary_struct.SecondaryStructure,
    'stability': stability.Stability,
    'fluorescence': fluorescence.Fluorescence,

    # Satellites.
    'eurosat': eurosat.EurosatPretrain,
    'eurosat_transfer': eurosat.EurosatTransfer,

    # Semiconductors.
    'wafer': wafer_map.WaferMapPretrain,
    'wafer_transfer': wafer_map.WaferMapTransfer,
}

PRETRAINING_DATASETS = [
    'cifar10_small', 'imagenet', 'pamap2', 'pamap2_spectrogram', 'mscoco', 'wikitext103', 'chexpert', 'librispeech', 'mc4',
    'genomics', 'higgs', 'pfam', 'wafer', 'eurosat'
]

UNLABELED_DATASETS = ['wikitext103', 'librispeech', 'mc4']
IGNORE_INDEX_DATASETS = {'secondary_struct': 3}
TOKENWISE_DATASETS = ['secondary_struct']

TRANSFER_DATASETS = [
    # Captioned images.
    'mismatched_caption',
    'vqa',

    # Medical images.
    'chexpert',
    'xray8',

    # Natural images (S).
    'aircraft_small',
    'cu_birds_small',
    'dtd_small',
    'traffic_sign_small',
    'vgg_flower_small',

    # Natural images (L).
    'cifar10',
    'aircraft',
    'cu_birds',
    'dtd',
    'traffic_sign',
    'vgg_flower',

    # Sensor.
    'pamap2',
    'pamap2_spectrogram',

    # Speech.
    'audio_mnist',
    'fluent_speech_location',
    'fluent_speech_action',
    'fluent_speech_object',
    'google_speech',
    'librispeech_transfer',
    'voxceleb1',

    # Text.
    'cola',
    'mnli_matched',
    'mnli_mismatched',
    'mrpc',
    'qnli',
    'qqp',
    'rte',
    'sst2',
    'stsb',
    'wnli',
    'paws_en',
    'paws_fr',
    'paws_es',
    'paws_de',
    'paws_zh',
    'paws_ja',
    'paws_ko',

    # Genomics.
    'genomics_transfer_id',
    'genomics_transfer_ood',

    # Particle physics.
    'higgs_transfer',

    # Proteins.
    'pfam_transfer',
    'scop',
    'secondary_struct',
    'stability',
    'fluorescence',

    # Satellites.
    'eurosat_transfer',

    # Semiconductors.
    'wafer_transfer'
]

# template for each domain namedtuple

DomainTuple = namedtuple('DomainTuple', ['pretrain', 'transfers'])

PRETRAIN_TO_TRANSFER_DICT = {
    'captioned_images':
        DomainTuple(
            pretrain='mscoco',
            transfers=[
                'mismatched_caption',
                'vqa',
            ],
        ),
    'genomics':
        DomainTuple(
            pretrain='genomics',
            transfers=[
                'genomics_transfer_id',
                'genomics_transfer_ood',
            ],
        ),
    'medical_images':
        DomainTuple(
            pretrain='chexpert',
            transfers=[
                'xray8',
            ],
        ),
    'natural_images':
        DomainTuple(pretrain='imagenet', transfers=[
            'cifar10',
            'aircraft',
            'cu_birds',
            'dtd',
            'traffic_sign',
            'vgg_flower',
        ]),
    'particle_physics':
        DomainTuple(
            pretrain='higgs',
            transfers=[
                'higgs_transfer',
            ],
        ),
    'proteins':
        DomainTuple(
            pretrain='pfam',
            transfers=[
                'pfam_transfer',
                'scop',
                'secondary_struct',
                'stability',
                'fluorescence',
            ],
        ),
    'satellite_images':
        DomainTuple(
            pretrain='eurosat',
            transfers=[
                'eurosat_transfer',
            ],
        ),
    'semiconductors':
        DomainTuple(
            pretrain='wafer',
            transfers=[
                'wafer_transfer',
            ],
        ),
    'sensor':
        DomainTuple(
            pretrain='pamap2',
            transfers=[
                'pamap2',
            ],
        ),
    'speech':
        DomainTuple(
            pretrain='librispeech',
            transfers=[
                'audio_mnist',
                'fluent_speech_location',
                'fluent_speech_action',
                'fluent_speech_object',
                'google_speech',
                'librispeech_transfer',
                'voxceleb1',
            ],
        ),
    'text':
        DomainTuple(
            pretrain='wikitext103',
            transfers=[
                'cola',
                'mnli_matched',
                'mnli_mismatched',
                'mrpc',
                'qnli',
                'qqp',
                'rte',
                'sst2',
                'stsb',
                'wnli',
            ],
        ),
    'multi_text':
        DomainTuple(
            pretrain='mc4',
            transfers=[
                'paws_en',
                'paws_fr',
                'paws_es',
                'paws_de',
                'paws_zh',
                'paws_ja',
                'paws_ko',
            ],
        ),
}
