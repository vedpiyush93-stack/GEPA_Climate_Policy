---
license: cc-by-4.0
dataset_info:
  features:
  - name: text
    dtype: string
  - name: annotation_agent
    dtype: int64
  - name: geography
    dtype: string
  - name: region
    dtype: string
  - name: translated
    dtype: bool
  - name: annotation_NZT
    dtype: int64
  - name: annotation_Reduction
    dtype: int64
  - name: annotation_Other
    dtype: int64
  splits:
  - name: train
    num_bytes: 2912069
    num_examples: 2610
  download_size: 1522649
  dataset_size: 2912069
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# National Climate Targets Training Dataset – Climate Policy Radar

A dataset of climate targets made by national governments in their laws, policies and UNFCCC submissions which has been used to train a classifier. Text was sourced from the [Climate Policy Radar database](https://app.climatepolicyradar.org).

We define a target as an aim to achieve a specific outcome, that is quantifiable and is given a deadline. 

This dataset distinguishes between different types of targets:

- **Reduction** (a.k.a. emissions reduction): a target referring to a reduction in greenhouse gas emissions, either economy-wide or for a sector.
- **Net zero**: a commitment to balance GHG emissions with removal, effectively reducing the net emissions to zero.
- **Other**: those that do not fit into the Reduction or Net Zero category but satisfy our definition of a target, e.g. renewable energy targets.

*IMPORTANT NOTE:* this dataset has been used to train a machine learning model, and **is not a list of all climate targets published by national governments**.

For more information on dataset creation, [see our paper](https://arxiv.org/abs/2404.02822).

## Dataset Description

This dataset includes 2,610 text passages containing 1,193 target mentions annotated in a multilabel setting: one text passage can be assigned to 0 or more target types. This breaks down as follows.

|               |   Number of passages |
|:--------------|--------:|
| NZT           |     203 |
| Reduction     |     359 |
| Other         |     631 |
| No Annotation |    1,584 |

It was annotated by 3 domain-experts with steps taken to ensure consistency by measuring inter-annotator agreement. Annotator `2` is a data scientist, with a combination of sampling negatives and errors caught during posthoc reviews.

All text is in English: the `translated` column describes whether it has been translated from another language using the Google Cloud Translation API. Further to the text and annotations, we also include characteristics of the documents we use to make equity calculations and anonymised assignment of annotations to annotators.

For more information on the dataset and its creation see **our paper TBA**.

## License

Our dataset is licensed as [CC by 4.0](https://creativecommons.org/licenses/by/4.0/).

Please read our [Terms of Use](https://app.climatepolicyradar.org/terms-of-use), including any specific terms relevant to commercial use. Contact partners@climatepolicyradar.org with any questions.

## Links


- [Paper](https://arxiv.org/abs/2404.02822)

## Citation

*Juhasz, M., Marchand, T., Melwani, R., Dutia, K., Goodenough, S., Pim, H., & Franks, H. (2024). Identifying Climate Targets in National Laws and Policies using Machine Learning. arXiv preprint arXiv:2404.02822.*

```
@misc{juhasz2024identifying,
      title={Identifying Climate Targets in National Laws and Policies using Machine Learning}, 
      author={Matyas Juhasz and Tina Marchand and Roshan Melwani and Kalyan Dutia and Sarah Goodenough and Harrison Pim and Henry Franks},
      year={2024},
      eprint={2404.02822},
      archivePrefix={arXiv},
      primaryClass={cs.CY}
}
```


## Authors & Contact

Climate Policy Radar team: Matyas Juhasz, Tina Marchand, Roshan Melwani, Kalyan Dutia, Sarah Goodenough, Harrison Pim, and Henry Franks.

https://climatepolicyradar.org