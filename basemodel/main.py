# -*- coding: utf-8 -*-

from ACF import train_acf
from PFMC import PFMC_main
from FDSA import FDSA_main
from ANAM import ANAM_main
from HARNN import HARNN_main
from Caser import Caser_main
from SASRec import SASRec_main

seed = 0
factor = 64
batch_size = 2048
# noise_len = 20

#'Kindle'!
# for data in ['Instant_Video', 'Amazon_App', 'Kindle', 'Clothing', 'Games', 'Grocery']:
for data in ['ml-1m']:
    # PFMC_main(data, factor, seed, batch_size, 5)
    # SASRec_main(data, factor, seed, batch_size)
    # Caser_main(data, factor, seed, batch_size)
    # FDSA_main(data, factor, seed, batch_size)
    train_acf(data, factor, seed, batch_size)
    # HARNN_main(data, factor, seed, batch_size)
    # ANAM_main(data, factor, seed, batch_size)
