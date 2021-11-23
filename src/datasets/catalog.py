from src.datasets.captioned_images import mscoco, vqa
from src.datasets.medical_images import chexpert, xray8
from src.datasets.natural_images import aircraft, cifar, cu_birds, dtd, imagenet, traffic_sign, vgg_flower
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
}

PRETRAINING_DATASETS = [
    'cifar10_small', 'imagenet', 'pamap2', 'pamap2_spectrogram', 'mscoco', 'wikitext103', 'chexpert', 'librispeech', 'vqa',
    'mc4'
]
UNLABELED_DATASETS = ['wikitext103', 'librispeech', 'mc4']
MULTILABEL_DATASETS = ['chexpert', 'vqa']

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
    'paws_ko'
]
