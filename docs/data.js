window.REPORT = {
 "classes": [
  "background",
  "Impacted",
  "Caries",
  "Periapical Lesion",
  "Deep Caries"
 ],
 "colors": {
  "Impacted": "#c98500",
  "Caries": "#e66767",
  "Periapical Lesion": "#199e70",
  "Deep Caries": "#9085e9"
 },
 "thresh": 0.5,
 "iou": 0.5,
 "epochs": [
  {
   "epoch": 0,
   "loss": 0.699,
   "map": 0.172,
   "map50": 0.276,
   "per_class": {
    "Impacted": 0.45,
    "Caries": 0.179,
    "Periapical Lesion": 0.005,
    "Deep Caries": 0.054
   }
  },
  {
   "epoch": 1,
   "loss": 0.434,
   "map": 0.242,
   "map50": 0.363,
   "per_class": {
    "Impacted": 0.489,
    "Caries": 0.178,
    "Periapical Lesion": 0.207,
    "Deep Caries": 0.091
   }
  },
  {
   "epoch": 2,
   "loss": 0.392,
   "map": 0.251,
   "map50": 0.384,
   "per_class": {
    "Impacted": 0.516,
    "Caries": 0.209,
    "Periapical Lesion": 0.098,
    "Deep Caries": 0.178
   }
  },
  {
   "epoch": 3,
   "loss": 0.359,
   "map": 0.28,
   "map50": 0.448,
   "per_class": {
    "Impacted": 0.522,
    "Caries": 0.185,
    "Periapical Lesion": 0.169,
    "Deep Caries": 0.246
   }
  },
  {
   "epoch": 4,
   "loss": 0.325,
   "map": 0.289,
   "map50": 0.437,
   "per_class": {
    "Impacted": 0.539,
    "Caries": 0.236,
    "Periapical Lesion": 0.185,
    "Deep Caries": 0.195
   }
  },
  {
   "epoch": 5,
   "loss": 0.291,
   "map": 0.316,
   "map50": 0.481,
   "per_class": {
    "Impacted": 0.547,
    "Caries": 0.249,
    "Periapical Lesion": 0.211,
    "Deep Caries": 0.259
   }
  },
  {
   "epoch": 6,
   "loss": 0.262,
   "map": 0.295,
   "map50": 0.454,
   "per_class": {
    "Impacted": 0.549,
    "Caries": 0.238,
    "Periapical Lesion": 0.144,
    "Deep Caries": 0.25
   }
  },
  {
   "epoch": 7,
   "loss": 0.225,
   "map": 0.278,
   "map50": 0.441,
   "per_class": {
    "Impacted": 0.519,
    "Caries": 0.215,
    "Periapical Lesion": 0.171,
    "Deep Caries": 0.209
   }
  },
  {
   "epoch": 8,
   "loss": 0.185,
   "map": 0.301,
   "map50": 0.444,
   "per_class": {
    "Impacted": 0.57,
    "Caries": 0.231,
    "Periapical Lesion": 0.17,
    "Deep Caries": 0.233
   }
  },
  {
   "epoch": 9,
   "loss": 0.166,
   "map": 0.293,
   "map50": 0.436,
   "per_class": {
    "Impacted": 0.553,
    "Caries": 0.247,
    "Periapical Lesion": 0.137,
    "Deep Caries": 0.233
   }
  }
 ],
 "best_epoch": 5,
 "best_map": 0.316,
 "per_class": {
  "Impacted": {
   "tp": 35,
   "fp": 12,
   "fn": 5,
   "gt": 40,
   "precision": 0.745,
   "recall": 0.875
  },
  "Caries": {
   "tp": 56,
   "fp": 134,
   "fn": 45,
   "gt": 101,
   "precision": 0.295,
   "recall": 0.554
  },
  "Periapical Lesion": {
   "tp": 3,
   "fp": 3,
   "fn": 6,
   "gt": 9,
   "precision": 0.5,
   "recall": 0.333
  },
  "Deep Caries": {
   "tp": 14,
   "fp": 19,
   "fn": 18,
   "gt": 32,
   "precision": 0.424,
   "recall": 0.438
  }
 },
 "confusion": [
  [
   0,
   12,
   120,
   2,
   13
  ],
  [
   3,
   35,
   1,
   0,
   1
  ],
  [
   40,
   0,
   56,
   1,
   4
  ],
  [
   2,
   0,
   3,
   3,
   1
  ],
  [
   8,
   0,
   10,
   0,
   14
  ]
 ],
 "showcase": [
  {
   "stem": "val_28",
   "w": 1400,
   "h": 733,
   "gt": [
    {
     "cls": "Impacted"
    },
    {
     "cls": "Impacted"
    },
    {
     "cls": "Impacted"
    },
    {
     "cls": "Impacted"
    },
    {
     "cls": "Periapical Lesion"
    },
    {
     "cls": "Caries"
    },
    {
     "cls": "Caries"
    },
    {
     "cls": "Caries"
    },
    {
     "cls": "Caries"
    },
    {
     "cls": "Deep Caries"
    },
    {
     "cls": "Periapical Lesion"
    }
   ],
   "pred": [
    {
     "cls": "Impacted",
     "score": 0.98
    },
    {
     "cls": "Impacted",
     "score": 0.98
    },
    {
     "cls": "Impacted",
     "score": 0.98
    },
    {
     "cls": "Impacted",
     "score": 0.96
    },
    {
     "cls": "Deep Caries",
     "score": 0.96
    },
    {
     "cls": "Caries",
     "score": 0.83
    },
    {
     "cls": "Caries",
     "score": 0.67
    },
    {
     "cls": "Caries",
     "score": 0.54
    },
    {
     "cls": "Periapical Lesion",
     "score": 0.53
    }
   ]
  },
  {
   "stem": "val_14",
   "w": 1400,
   "h": 675,
   "gt": [
    {
     "cls": "Caries"
    },
    {
     "cls": "Caries"
    },
    {
     "cls": "Deep Caries"
    },
    {
     "cls": "Impacted"
    },
    {
     "cls": "Periapical Lesion"
    }
   ],
   "pred": [
    {
     "cls": "Caries",
     "score": 0.92
    },
    {
     "cls": "Deep Caries",
     "score": 0.92
    },
    {
     "cls": "Caries",
     "score": 0.92
    },
    {
     "cls": "Caries",
     "score": 0.65
    },
    {
     "cls": "Deep Caries",
     "score": 0.65
    },
    {
     "cls": "Caries",
     "score": 0.65
    },
    {
     "cls": "Impacted",
     "score": 0.57
    }
   ]
  },
  {
   "stem": "val_21",
   "w": 1400,
   "h": 644,
   "gt": [
    {
     "cls": "Caries"
    },
    {
     "cls": "Deep Caries"
    },
    {
     "cls": "Caries"
    },
    {
     "cls": "Impacted"
    },
    {
     "cls": "Caries"
    },
    {
     "cls": "Impacted"
    }
   ],
   "pred": [
    {
     "cls": "Impacted",
     "score": 0.9
    },
    {
     "cls": "Impacted",
     "score": 0.84
    },
    {
     "cls": "Periapical Lesion",
     "score": 0.72
    },
    {
     "cls": "Caries",
     "score": 0.64
    },
    {
     "cls": "Caries",
     "score": 0.59
    },
    {
     "cls": "Caries",
     "score": 0.58
    },
    {
     "cls": "Deep Caries",
     "score": 0.57
    }
   ]
  },
  {
   "stem": "val_41",
   "w": 1400,
   "h": 734,
   "gt": [
    {
     "cls": "Deep Caries"
    },
    {
     "cls": "Deep Caries"
    },
    {
     "cls": "Impacted"
    },
    {
     "cls": "Caries"
    },
    {
     "cls": "Impacted"
    },
    {
     "cls": "Caries"
    }
   ],
   "pred": [
    {
     "cls": "Deep Caries",
     "score": 0.95
    },
    {
     "cls": "Caries",
     "score": 0.78
    },
    {
     "cls": "Caries",
     "score": 0.78
    },
    {
     "cls": "Impacted",
     "score": 0.58
    },
    {
     "cls": "Caries",
     "score": 0.51
    },
    {
     "cls": "Deep Caries",
     "score": 0.51
    }
   ]
  },
  {
   "stem": "val_30",
   "w": 1400,
   "h": 733,
   "gt": [
    {
     "cls": "Periapical Lesion"
    },
    {
     "cls": "Periapical Lesion"
    },
    {
     "cls": "Caries"
    },
    {
     "cls": "Deep Caries"
    },
    {
     "cls": "Deep Caries"
    }
   ],
   "pred": [
    {
     "cls": "Periapical Lesion",
     "score": 0.9
    },
    {
     "cls": "Periapical Lesion",
     "score": 0.87
    },
    {
     "cls": "Periapical Lesion",
     "score": 0.76
    },
    {
     "cls": "Periapical Lesion",
     "score": 0.72
    }
   ]
  },
  {
   "stem": "val_18",
   "w": 1400,
   "h": 613,
   "gt": [
    {
     "cls": "Caries"
    },
    {
     "cls": "Deep Caries"
    },
    {
     "cls": "Periapical Lesion"
    }
   ],
   "pred": [
    {
     "cls": "Caries",
     "score": 0.75
    },
    {
     "cls": "Caries",
     "score": 0.73
    },
    {
     "cls": "Caries",
     "score": 0.65
    },
    {
     "cls": "Caries",
     "score": 0.58
    }
   ]
  }
 ],
 "n_val_images": 50
};
