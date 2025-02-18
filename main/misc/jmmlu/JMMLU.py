# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import datasets
import pandas as pd


_CITATION = """\
@misc{yin2024respect,
      title={Should We Respect LLMs? A Cross-Lingual Study on the Influence of Prompt Politeness on LLM Performance}, 
      author={Ziqi Yin and Hao Wang and Kaito Horio and Daisuke Kawahara and Satoshi Sekine},
      year={2024},
      eprint={2402.14531},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
JMMLU is a four-choice question set consisting of Japanese-translated questions of a portion of MMLU (Translated questions) and questions based on unique Japanese cultural context (Japanese questions). It is designed to assess the performance of large language models in Japanese. JMMLU consists of 7,536 questions in the following 56 tasks (subjects).
"""

_LICENSE = "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License"

_URL = r"https://huggingface.co/datasets/nlp-waseda/JMMLU/resolve/main/JMMLU.zip"

task_list = [
'japanese_history', 
'miscellaneous', 
'security_studies', 
'virology', 
'nutrition', 
'human_sexuality', 
'college_mathematics', 
'japanese_civics', 
'econometrics', 
'computer_security', 
'clinical_knowledge', 
'machine_learning', 
'high_school_chemistry', 
'human_aging', 
'logical_fallacies', 
'sociology', 
'high_school_european_history', 
'high_school_statistics', 
'high_school_physics', 
'high_school_microeconomics', 
'college_physics', 
'anatomy', 
'high_school_psychology', 
'business_ethics', 
'professional_psychology', 
'college_medicine', 
'elementary_mathematics', 
'moral_disputes', 
'marketing', 
'high_school_macroeconomics', 
'world_religions',
'conceptual_physics',
'professional_medicine',
'prehistory',
'high_school_mathematics',
'international_law',
'philosophy',
'japanese_idiom',
'japanese_geography',
'management',
'high_school_computer_science',
'medical_genetics',
'college_computer_science',
'public_relations',
'professional_accounting',
'abstract_algebra',
'global_facts',
'college_biology',
'high_school_geography',
'world_history',
'high_school_biology',
'college_chemistry',
'electrical_engineering',
'astronomy',
'jurisprudence',
'formal_logic']


class JMMLUConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class JMMLU(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        JMMLUConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "JMMLU","test", f"{task_name}.csv"
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath,encoding="utf-8-sig")
        for i, instance in enumerate(df.to_dict(orient="records")):
            yield i, instance