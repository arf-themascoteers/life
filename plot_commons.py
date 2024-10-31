ALGS = {
    "bnc": "BS-Net-Classifier [9]",
    "c1": "BS-DSC",
    "all": "All Bands",
    "mcuve": "MCUVE [17]",
    "bsnet": "BS-Net-FC [2]",
    "pcal": "PCAL [16]",
    "bsdr": "BSDR",
    "bsdrattn": "BSDR-ATTN",
    "c1_wo_dsc": "BS-DSC-EXCL",
    "msobsdr": "MSO-BSDR",
    "linspacer": "Linearly Spaced",
    "random": "Randomly Selected",
}

FIXED_ALG_COLORS = {
    "bnc": "#1f77b4",
    "c1": "#d62728",
    "all": "#2ca02c",
    "mcuve": "#ff7f0e",
    "bsnet": "#008000",
    "pcal": "#9467bd",
    "bsdr": "#7FFF00",
    "bsdrattn": "#7F0000",
    "linspacer": "#FF00FF",
    "random": "#d6ff28",
    "c1_wo_dsc": "#bcbd22",
    "msobsdr": "#17becf"
}

ARBITRARY_ALG_COLORS = ["#000000","#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MARKERS = ['s', 'P', 'D', '^', 'o', '*', '.', 's', 'P', 'D', '^', 'o', '*', '.']
ALG_ORDERS = ["all", "random", "linspacer", "pcal", "mcuve", "bsnet", "bnc", "c1", "bsdr","bsdrattn"]