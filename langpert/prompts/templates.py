"""
Prompt templates for different biological contexts.
"""

# Default general-purpose prompt
DEFAULT_PROMPT = r'''
Instruction: Analyze the gene {gene} and identify {k_range} most similar genes from the provided list. Rank them by similarity (most similar first).

Consider similarity based on:
- Shared biological pathways
- Co-regulation patterns
- Similar protein-protein interactions
- Similar effects when knocked out

Available genes: {list_of_genes}

Format your response as JSON with two parts:
1. "reasoning": Explain your analysis, discussing potential connections between {gene} and relevant genes
2. "kNN": List the most similar genes in order of similarity

Example response format:
{{
  "reasoning": "Gene X is involved in pathway Y which directly interacts with gene Z...",
  "kNN": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
}}

DO NOT provide multiple JSON objects or alternative analyses. Provide ONLY ONE response.
'''

# Minimal prompt
MINIMAL_PROMPT = r'''
Instruction: From the provided list, identify the {k_range} genes most similar to gene {gene}.

Available genes: {list_of_genes}

Format your response as JSON:
{{
  "reasoning": "Brief explanation",
  "kNN": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
}}
'''

# No reasoning version
NO_REASONING_PROMPT = r'''
Instruction: Identify the {k_range} genes most similar to gene {gene} from the provided list.

Available genes: {list_of_genes}

Format your response as JSON containing only the ranked list:
{{
  "kNN": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
}}

Do not include explanations, reasoning, or any other fields. Provide only the JSON object with the "kNN" field.
'''

# Cell-type specific prompt (K562)
K562_PROMPT = r'''
Instruction: Analyze the gene {gene} and identify {k_range} most similar genes from the provided list. Rank them by similarity (most similar first).

Consider similarity based on:
- Shared biological pathways
- Co-regulation patterns
- Similar protein-protein interactions
- Similar effects when knocked out

Context: Analysis should focus on the K562 cell line (chronic myeloid leukemia model). Consider cancer-relevant pathways including ribosome biogenesis, transcriptional regulation, mitochondrial function, and stress responses.

Available genes: {list_of_genes}

Format your response as JSON with two parts:
1. "reasoning": Explain your analysis, discussing potential connections between {gene} and relevant genes
2. "kNN": List the most similar genes in order of similarity

Example response format:
{{
  "reasoning": "Gene X is involved in pathway Y which directly interacts with gene Z...",
  "kNN": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
}}

DO NOT provide multiple JSON objects or alternative analyses. Provide ONLY ONE response.
'''
#  Consider cancer-relevant pathways including {related_pathways}.
CL_PROMPT = r'''
Instruction: Analyze the gene {gene} and identify {k_range} most similar genes from the provided list. Rank them by similarity (most similar first).

Consider similarity based on:
- Shared biological pathways
- Co-regulation patterns
- Similar protein-protein interactions
- Similar effects when knocked out

Context: Analysis should focus on the {cell_line} cell line.

Available genes: {list_of_genes}

Format your response as JSON with two parts:
1. "reasoning": Explain your analysis, discussing potential connections between {gene} and relevant genes
2. "kNN": List the most similar genes in order of similarity

Example response format:
{{
  "reasoning": "Gene X is involved in pathway Y which directly interacts with gene Z...",
  "kNN": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
}}

DO NOT provide multiple JSON objects or alternative analyses. Provide ONLY ONE response.
'''

REFINE_CL_PROMPT = r'''
As an expert in gene perturbation prediction for the {cell_line} cell line, your task is to carefully review and, if necessary, alter the following LIST {single_pass_gene_list} based on their relevance to perturbation prediction and similarity to the {gene} gene of interest. 

Available genes: {list_of_genes}

Consider the biological pathways, co-regulation, and protein-protein interactions of each gene. Ensure that the listed genes are highly relevant for perturbation prediction and are likely to result in similar changes in gene expression as the gene of interest when perturbed. You may replace or remove genes as needed to optimize the list for perturbation prediction. Please make any necessary alterations to the gene list to improve its relevance for perturbation prediction in the context of {cell_line} cell line. 

Format your response as JSON with two parts:
1. "reasoning": Explain your analysis, discussing potential connections between {gene} and relevant genes
2. "kNN": List the most similar genes in order of similarity

Example response format:
{{
  "reasoning": "Gene X is involved in pathway Y which directly interacts with gene Z...",
  "kNN": ["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"]
}}

Once you have reviewed and made any alterations, provide the updated response. DO NOT provide multiple JSON objects or alternative analyses. Provide ONLY ONE response.
'''

CL_DRUG_PROMPT = r'''
Instruction: Analyze the drug {drug} and identify {k_range} most similar drugs from the provided list. Rank them by similarity (most similar first).

Consider similarity based on:
- Shared biological pathways
- Co-regulation patterns
- Similar protein-protein interactions
- Similar effects when knocked out

Context: Analysis should focus on the {cell_line} cell line.

Available drugs: {list_of_drugs}

Format your response as JSON with two parts:
1. "reasoning": Explain your analysis, discussing potential connections between {drug} and relevant drugs
2. "kNN": List the most similar drugs in order of similarity

Example response format:
{{
  "reasoning": "Drug X is involved in pathway Y which directly interacts with drug Z...",
  "kNN": ["Drug1", "Drug2", "Drug3", "Drug4", "Drug5"]
}}

DO NOT provide multiple JSON objects or alternative analyses. Provide ONLY ONE response.
'''

REFINE_CL_DRUG_PROMPT = r'''
As an expert in drug perturbation prediction for the {cell_line} cell line, your task is to carefully review and, if necessary, alter the following LIST {single_pass_drug_list} based on their relevance to perturbation prediction and similarity to the {drug} drug of interest. 

Available drugs: {list_of_drugs}

Consider the biological pathways, co-regulation, and protein-protein interactions of each drug. Ensure that the listed drugs are highly relevant for perturbation prediction and are likely to result in similar changes in gene expression as the drug of interest when perturbed. You may replace or remove drugs as needed to optimize the list for perturbation prediction. Please make any necessary alterations to the drug list to improve its relevance for perturbation prediction in the context of {cell_line} cell line. 

Format your response as JSON with two parts:
1. "reasoning": Explain your analysis, discussing potential connections between {drug} and relevant drugs
2. "kNN": List the most similar drugs in order of similarity

Example response format:
{{
  "reasoning": "Drug X is involved in pathway Y which directly interacts with drug Z...",
  "kNN": ["Drug1", "Drug2", "Drug3", "Drug4", "Drug5"]
}}

Once you have reviewed and made any alterations, provide the updated response. DO NOT provide multiple JSON objects or alternative analyses. Provide ONLY ONE response.
'''

# Template registry
PROMPT_TEMPLATES = {
    "default": DEFAULT_PROMPT,
    "minimal": MINIMAL_PROMPT,
    "no_reasoning": NO_REASONING_PROMPT,
    "k562": K562_PROMPT,
    "cell_line_specific": CL_PROMPT,
    "cell_line_refine": REFINE_CL_PROMPT,
    "cell_line_drug_specific": CL_DRUG_PROMPT,
    "cell_line_drug_refine": REFINE_CL_DRUG_PROMPT,
}