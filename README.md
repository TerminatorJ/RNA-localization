# RNA-localization
This repository can be used to data sharing, code sharing, and feedbacks

Descriptions about the files.

you can treat early_fusion and sequential as two wings of the project.
"early fusion" is a wing that repeat the DM3Loc and do early fusion of one-hot encoding and parnet output with some kinds of variants
"sequential" is able to make the DM3Loc and parnet run in parallelly.
booster is the main control scripts of the model, where you can control whether to run them individually and change the input dimentions, doing the grid search.


install the Viennarna to get the RNA structure information

```linux
conda install -c bioconda viennarna
```

