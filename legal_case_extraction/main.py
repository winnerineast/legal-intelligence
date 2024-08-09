import os
import fitz  # PyMuPDF
import itertools
import re
import json
import yaml
import numpy as np
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI
from typing import Dict, List, Optional, Tuple, Union


class VectorDatabase:
    def __init__(self, dimension: int) -> None:
        self.dimension: int = dimension
        self.data: Dict[str, np.ndarray] = {}

    def add_vector(self, key: str, vector: List[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension ({len(vector)}) does not match specified dimension ({self.dimension}).")

        self.data[key] = np.array(vector)

    def get_vector(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve a vector from the database.
        """
        return self.data.get(key, None)

    def remove_vector(self, key: str) -> None:
        if key in self.data:
            del self.data[key]

    def list_keys(self) -> List[str]:
        return list(self.data.keys())

    def cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> Optional[float]:
        if vector1 is None or vector2 is None:
            return None

        dot_product: float = np.dot(vector1, vector2)
        norm1: float = np.linalg.norm(vector1)
        norm2: float = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            return None

        similarity: float = dot_product / (norm1 * norm2)
        return similarity.item()

    def similarity(self, key1: str, key2: str) -> Optional[float]:
        """
        Compute cosine similarity between two vectors in the database.
        """
        vector1: Optional[np.ndarray] = self.data.get(key1, None)
        vector2: Optional[np.ndarray] = self.data.get(key2, None)

        similarity = self.cosine_similarity(vector1, vector2)
        return similarity

    def search(self, query_vector: np.ndarray, top_k: int = 1):
        """
        Search for the closest vectors to a given query vector based on cosine similarity,
        returning the top `k` closest keys.
        """
        if not self.data:
            return []

        # Collect all similarities
        similarities = []
        for key, vector in self.data.items():
            similarity = self.cosine_similarity(query_vector, vector)
            similarities.append((key, similarity))

        # Sort once by similarity in descending order and select the top_k elements
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_keys = similarities[:top_k]

        return [key for key, _ in top_k_keys]

    def find_similar_pairs(self, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[
        Tuple[str, str, float]]:
        """
        Find the most similar pairs of vectors in the database.
        If a threshold is given, only pairs with similarity above this value are considered.
        If a top_k value is given, only the top k most similar pairs are returned.
        At least one of the parameters (top_k or threshold) must be provided.
        """
        if top_k is None and threshold is None:
            raise ValueError("At least one of 'top_k' or 'threshold' must be provided")

        keys = list(self.data.keys())
        pair_similarities = []

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                sim = self.cosine_similarity(self.data[keys[i]], self.data[keys[j]])
                if sim is not None and (threshold is None or sim > threshold):
                    pair_similarities.append((keys[i], keys[j], sim))

        pair_similarities.sort(key=lambda x: x[2], reverse=True)

        # If top_k is not specified, return all pairs above the threshold
        if top_k is None:
            return pair_similarities
        else:
            return pair_similarities[:top_k]

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __getitem__(self, key: str) -> Optional[np.ndarray]:
        return self.get_vector(key)

    def __setitem__(self, key: str, vector: List[float]) -> None:
        self.add_vector(key, vector)

    def __delitem__(self, key: str) -> None:
        self.delete_vector(key)

    def __iter__(self):
        return iter(self.data)

    def save_vectors(self, filepath: str) -> None:
        np.savez(filepath, **self.data)


class LegalCaseAgent:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", configs: dict = None):
        self.model_name = model_name
        self.llm_client = OpenAI(
            api_key=configs['api-key'][0],
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.embedding_model = OpenAI(
            api_key=configs['api-key'][0],
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.chunks = None
        login(token=configs['api-key'][1])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.configs = configs

        # Data
        self.fact_global_pointers: list = None
        self.law_global_pointers: list = None

        self.chunks: list[dict] = None

        # Load prompts
        self.prompts = self.load_prompts()

    def load_prompts(self):
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        prompt_template_names = [
            "summary",
            "facts_extraction",
            "law_extraction",
            "decision_extraction",
            "relation_extraction",
            "reduce_summary",
            "deduplication",
            "match_missing_refs"
        ]

        prompt_templates = {}
        for prompt_template in prompt_template_names:
            with open(os.path.join(root_path, f"{prompt_template}.txt"), 'r') as f:
                prompt_templates[prompt_template] = f.read()

        return prompt_templates

    def token_length(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False))

    def set_raw_document(self, raw_document: str):
        self.raw_document = raw_document

    def chat(self, *args, **kwargs):
        # Default parameters
        if "model" not in kwargs:
            kwargs["model"] = self.model_name

        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.0

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4196

        chat_completion = self.llm_client.chat.completions.create(
            *args, **kwargs
        )
        return chat_completion.choices[0].message.content

    def get_embedding(self, string: str, client: OpenAI):
        def normalize_l2(x):
            x = np.array(x)
            if x.ndim == 1:
                norm = np.linalg.norm(x)
                if norm == 0:
                    return x
                return x / norm
            else:
                norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
                return np.where(norm == 0, x, x / norm)

        response = client.embeddings.create(
            model="BAAI/bge-base-en-v1.5",
            input=string,
            encoding_format="float"
        )
        embedding = response.data[0].embedding
        normalized_embedding = normalize_l2(embedding)

        return normalized_embedding

    def split(self, raw_document) -> list[str]:
        lines = raw_document.replace('\r', '\n').split('\n')
        chunks = []
        cum_tokens = 0
        section = []
        for line in lines:
            if not line:
                continue

            line_token_length = self.token_length(line)
            if line_token_length > self.configs['max-chunk-tokens']:
                raise ValueError(f"Line too long ({line_token_length} tokens)")

            cum_tokens += line_token_length
            if cum_tokens <= self.configs['max-chunk-tokens']:
                section.append(line)
            else:
                chunks.append('\n'.join(section))
                section = [line]
                cum_tokens = self.token_length(line)
        chunks.append('\n'.join(section))

        chunks = {f"C{i}": dict(content=chunk) for i, chunk in enumerate(chunks)}

        return chunks

    def summarize(self, input_text):
        summary = self.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': self.prompts['summary']
                },
                {
                    'role': 'user',
                    'content': input_text
                }
            ]
        )
        return summary

    def reduce_summary(self, summaries):
        summaries_text = ""
        for i, summary in enumerate(summaries, start=1):
            summaries_text += f"Section {i}:\n{summary}\n\n"

        return self.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': self.prompts['reduce_summary']
                },
                {
                    'role': 'user',
                    'content': summaries_text
                }
            ]
        )

    def extract_facts(self, chunk: str, chat_func):
        prompt = ""
        sys_prompt = self.prompts['facts_extraction']
        if self.configs["include-summary-in-prompt"]:
            sys_prompt += (
                    "## Case Summary\n" +
                    self.summary + "\n\n"
            )
        prompt += "## Text Segments\n" + chunk

        raw_response = chat_func(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': sys_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        # Extract markdown
        facts = extract_json_from_markdown(raw_response)
        if facts is None:
            return []

        # Rename "summarization" to "content"
        for fact in facts:
            fact["content"] = fact["summarization"]
            del fact["summarization"]

        # Filter facts by post-classification results
        def fact_filter_criteria(fact):
            if fact["is_in_court"]:
                return False
            if fact["is_terms_or_contracts_between_plaintiff_or_defendant"]:
                return True
            if fact["is_law_related"]:
                return False
            return True

        facts = [fact for fact in facts if fact_filter_criteria(fact)]

        return facts

    def extract_laws(self, chunk: str, chat_func):
        prompt = ""
        sys_prompt = self.prompts['law_extraction']
        if self.configs["include-summary-in-prompt"]:
            sys_prompt += (
                    "## Case Summary\n" +
                    self.summary + "\n\n"
            )
        prompt += "## Text Segments\n" + chunk

        raw_response = chat_func(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': sys_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        laws = extract_json_from_markdown(raw_response)
        if laws is None:
            return []

        # Filter laws by post-classification results
        def law_filter_criteria(law):
            return not law["is_from_the_same_case"]

        laws = [law for law in laws if law_filter_criteria(law)]
        return laws

    def extract_decisions(self, chunk: str, chat_func):
        prompt = ""
        sys_prompt = self.prompts['decision_extraction']
        if self.configs["include-summary-in-prompt"]:
            sys_prompt += (
                    "## Case Summary\n" +
                    self.summary + "\n\n"
            )
        prompt += "## Text Segments\n" + chunk

        raw_response = chat_func(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': sys_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        decisions = extract_json_from_markdown(raw_response)
        if decisions is None:
            return []

        return decisions

    def extract_relations(
            self,
            chunk: str,
            laws: dict,
            facts: dict,
            decisions: dict,
            chat_func
    ) -> list[dict]:
        prompt = ""
        sys_prompt = self.prompts['relation_extraction']
        if self.configs["include-summary-in-prompt"]:
            sys_prompt += (
                    "## Case Summary\n" +
                    self.summary + "\n\n"
            )

        prompt += (
            "## The paragraph\n"
            "```\n{chunk}\n```\n\n"
            "## Laws\n"
            "{laws}\n\n"
            "## Facts\n"
            "{facts}\n\n"
            "## Court Decisions\n"
            "{decisions}\n\n"
        )
        # Prepare prompts
        laws_str = "\n".join([f"{key}. {value}" for key, value in laws.items()])
        facts_str = "\n".join([f"{key}. {value}" for key, value in facts.items()])
        decisions_str = "\n".join([f"{key}. {value}" for key, value in decisions.items()])
        user_message = prompt.format(
            chunk=chunk,
            laws=laws_str,
            facts=facts_str,
            decisions=decisions_str
        )

        raw_response = chat_func(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': sys_prompt
                },
                {
                    'role': 'user',
                    'content': user_message
                }
            ]
        )

        relations = extract_json_from_markdown(raw_response)
        if relations is None:
            return []

        for relation in relations:
            relation["content"] = relation["relation_summary"]

        return relations

    def preprocess_doc(self, raw_document):
        chunks = self.split(raw_document)
        self.laws = {}
        self.facts = {}
        self.decisions = {}
        self.relations = []

        law_id = 1
        fact_id = 1
        decision_id = 1

        # First pass, for summarization
        for chunk_id, chunk_node in tqdm(chunks.items(), desc="Summarizing chunks..."):
            chunk_node['summary'] = self.summarize(chunk_node['content'])
            # chunk_node['embedding'] = self.get_embedding(chunk_node['content'], self.embedding_model)

        self.summary = self.reduce_summary(chunk["summary"] for chunk in chunks.values())

        # Second pass, for extraction
        for chunk_id, chunk_node in tqdm(chunks.items(), desc="Extracting items..."):
            # Extract laws
            laws = self.extract_laws(chunk_node['content'], self.chat)
            for law in laws:
                law["chunk_id"] = chunk_id
                key = f"L{law_id}"
                self.laws[key] = law
                law_id += 1

            # Extract facts
            facts = self.extract_facts(chunk_node['content'], self.chat)
            for fact in facts:
                fact["chunk_id"] = chunk_id
                key = f"F{fact_id}"
                self.facts[key] = fact
                fact_id += 1

            # Extract decisions
            decisions = self.extract_decisions(chunk_node['content'], self.chat)
            for decision in decisions:
                decision["chunk_id"] = chunk_id
                key = f"D{decision_id}"
                self.decisions[key] = decision
                decision_id += 1

            # Extract contexts within the current chunk, and extract relations
            relations = self.extract_relations(
                chunk=chunk_node['content'],
                laws={key: law["content"] for key, law in self.laws.items() if law["chunk_id"] == chunk_id},
                facts={key: fact["content"] for key, fact in self.facts.items() if fact["chunk_id"] == chunk_id},
                decisions={key: decision["content"] for key, decision in self.decisions.items() if
                           decision["chunk_id"] == chunk_id},
                chat_func=self.chat
            )
            self.relations.extend(relations)

        self.chunks = chunks

    def deduplicate_items(self):
        self.item_vector_store = VectorDatabase(dimension=self.configs["embedder-dim"])
        # Save embeddings of items
        total = len(self.facts) + len(self.laws) + len(self.decisions)
        for item_id, item in tqdm(itertools.chain(
                self.facts.items(),
                self.laws.items(),
                self.decisions.items()
        ), desc="Embedding items...", total=total):
            item_vector = self.get_embedding(item["content"], self.embedding_model)
            self.item_vector_store.add_vector(item_id, item_vector)

        self.item_vector_store.save_vectors("item_vectors.npz")
        similar_pairs = self.item_vector_store.find_similar_pairs(
            top_k=self.configs["deduplication"]["top-k-similar-items"],
            threshold=self.configs["deduplication"]["similarity-threshold"]
        )

        print(f"Found {len(similar_pairs)} similar pairs:")
        print(similar_pairs)

        substituted_pairs = {}
        for item1_id, item2_id, similarity in similar_pairs:
            # Check if the items are already substituted.
            # This may happen if an item appears more than once in the list of similar pairs.
            possible_substituted_item1 = item1_id
            while possible_substituted_item1 in substituted_pairs:
                possible_substituted_item1 = substituted_pairs[possible_substituted_item1]
            possible_substituted_item2 = item2_id
            while possible_substituted_item2 in substituted_pairs:
                possible_substituted_item2 = substituted_pairs[possible_substituted_item2]
            item1 = self.get_item(possible_substituted_item1)
            item2 = self.get_item(possible_substituted_item2)

            # Remove self-relations
            self.relations = [relation for relation in self.relations if relation["id1"] != relation["id2"]]
            # Skip if the items are the same
            if possible_substituted_item1 == possible_substituted_item2:
                continue

            print(
                f"Checking similarity between {possible_substituted_item1} and {possible_substituted_item2}, similarity {similarity}...")
            is_duplicate, data_to_keep = self.identify_duplication(item1, item2, self.chat)
            if not is_duplicate:
                continue

            item_id_to_keep = data_to_keep.strip()
            item_id_to_remove = None
            if data_to_keep == "Item 1":
                item_id_to_keep = possible_substituted_item1
                item_id_to_remove = possible_substituted_item2
            elif data_to_keep == "Item 2":
                item_id_to_keep = possible_substituted_item2
                item_id_to_remove = possible_substituted_item1
            else:
                print(f"Invalid `data_to_keep` in response: '{data_to_keep}'. Skipping.")
                continue

            print(f"Identified duplication: {item_id_to_keep} and {item_id_to_remove}. Keeping {data_to_keep}.")

            self.remove_item(item_id_to_remove)
            self.item_vector_store.remove_vector(item_id_to_remove)

            # Relink relations involving the removed item to the kept item
            for relation in self.relations:
                if relation["id1"] == item_id_to_remove:
                    relation["id1"] = item_id_to_keep
                if relation["id2"] == item_id_to_remove:
                    relation["id2"] = item_id_to_keep

            substituted_pairs[item_id_to_remove] = item_id_to_keep

    def identify_duplication(self, item1, item2, chat_func):
        prompt = ""
        sys_prompt = self.prompts['deduplication']
        if self.configs["include-summary-in-prompt"]:
            sys_prompt += (
                    "## Case Summary\n" +
                    self.summary + "\n\n"
            )
        # prompt += "## Text Segments\n"
        # prompt += f"### Item 1 Related\n{self.chunks[item1['chunk_id']]["content"]}\n\n"
        # prompt += f"### Item 2 Related\n{self.chunks[item2['chunk_id']]["content"]}\n\n"

        prompt += (
            "## Item 1\n"
            f"{item1['content']}\n\n"
            "## Item 2\n"
            f"{item2['content']}\n\n"
        )

        raw_response = chat_func(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': sys_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        json_data = extract_json_from_markdown(raw_response)
        if json_data is None or "is_duplicate" not in json_data:
            print("Wrong response format. The response is:")
            print(raw_response)
            return False, None

        if json_data["is_duplicate"]:
            return True, json_data["data_to_keep"]
        else:
            return False, None

    def fill_missing_relations(self):
        new_relations = []
        for relation in self.relations:
            if relation["id2"] is None:
                query = relation["description_of_item2_if_missing"]
                if query is None:
                    print("Missing query for missing item. Skipping.")
                    continue
                query_vector = self.get_embedding(query, self.embedding_model)
                item_ids = self.item_vector_store.search(query_vector,
                                                         top_k=self.configs["fill-missing-relations-top-k"])
                candidates = [(item_id, self.get_item(item_id)) for item_id in item_ids]
                # Query LLM to match the missing item
                is_matched, matched_item_id = self.identify_missing_relation(query, candidates, self.chat)
                if not is_matched:
                    print("No match found. Skipping.")
                    continue
                relation["id2"] = matched_item_id
                print(f"Matched missing item: {matched_item_id}")
                new_relations.append(relation)

        self.relations = new_relations

    def identify_missing_relation(self, query, candidates, chat_func):
        print(query)
        print(candidates)
        prompt = ""
        sys_prompt = self.prompts['match_missing_refs']
        if self.configs["include-summary-in-prompt"]:
            sys_prompt += (
                    "## Case Summary\n" +
                    self.summary + "\n\n"
            )
        # prompt += "## Text Segments\n"
        # prompt += f"### Item 1 Related\n{self.chunks[item1['chunk_id']]["content"]}\n\n"
        # prompt += f"### Item 2 Related\n{self.chunks[item2['chunk_id']]["content"]}\n\n"

        prompt += (
                "## Description of Missing item\n" +
                query
        )
        prompt += "\n\n## Candidate items\n"
        for item_id, candidate in candidates:
            prompt += f"\"{item_id}\": {candidate['content']}\n"

        raw_response = chat_func(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': sys_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        print(raw_response)
        json_data = extract_json_from_markdown(raw_response)

        # Parse results
        if json_data["is_query_matched"]:
            return True, json_data["matched_item_id"]
        else:
            return False, None

    def get_item(self, item_id):
        if item_id[0] == "F":
            return self.facts[item_id]
        elif item_id[0] == "L":
            return self.laws[item_id]
        elif item_id[0] == "D":
            return self.decisions[item_id]
        else:
            raise ValueError(f"Invalid item id: {item_id}")

    def remove_item(self, item_id):
        if item_id.startswith('F'):
            del self.facts[item_id]
        elif item_id.startswith('L'):
            del self.laws[item_id]
        elif item_id.startswith('D'):
            del self.decisions[item_id]
        else:
            raise ValueError(f"Invalid item id: {item_id}")

    def add_item(self, item_id, chunk_id, item):
        item["chunk_id"] = chunk_id
        if item_id.startswith('F'):
            self.facts[item_id] = item
        elif item_id.startswith('L'):
            self.laws[item_id] = item
        elif item_id.startswith('D'):
            self.decisions[item_id] = item
        else:
            raise ValueError(f"Invalid item id: {item_id}")

    def parse_document(self, doc: str):
        self.preprocess_doc(doc)
        self.deduplicate_items()
        self.fill_missing_relations()
        print("Document parsed successfully.")

    def save_state(self, path):
        """
        Save the current state of the legal case agent to a file.
        """
        state = {
            'chunks': self.chunks,
            'summary': self.summary,
            'facts': self.facts,
            'laws': self.laws,
            'decisions': self.decisions,
            'relations': self.relations,
        }
        with open(path, 'w') as file:
            json.dump(state, file, indent=4)

    def load_state(self, path):
        """
        Load the state of the legal case agent from a file.
        """
        with open(path, 'r') as file:
            state = json.load(file)
        self.chunks = state['chunks']
        self.summary = state['summary']
        self.facts = state['facts']
        self.laws = state['laws']
        self.decisions = state['decisions']
        self.relations = state['relations']

    def save_for_visualization(self, path):
        """
        Save the preprocessed document to a file in a format that can be visualized
        """
        data_dict = {
            "chunks": self.chunks,
            "facts": self.facts,
            "laws": self.laws,
            "decisions": self.decisions,
            "relations": self.relations
        }

        with open(path, 'w') as f:
            json.dump(data_dict, f, indent=4)


def try_fix_json(json_str: str) -> str:
    """
    Try to fix a JSON string by replacing double quotes in the values with single quotes.
    """
    lines = json_str.split("\n")
    new_lines = []
    for line in lines:
        num_quotes = line.count('"')
        if num_quotes > 4:
            # print(line)
            pattern = r'\"(.*?)\"\s*:\s*\"(.*)\".*'
            match = re.search(pattern, line)
            start, end = match.span(2)
            value_with_single_quotes = match.group(2).replace('"', "'")
            new_line = line[:start] + value_with_single_quotes + line[end:]

            new_lines.append(new_line)
        else:
            new_lines.append(line)

    fixed_json = "\n".join(new_lines)
    return fixed_json


def extract_json_from_markdown(text: str) -> list[str]:
    # Extract json block using regex
    pattern = "```\n?(?:json)?(.*?)```"
    match = re.findall(pattern, text, re.DOTALL)
    if match:
        potential_json = match[0]
    else:
        # Fall back to parse all text
        potential_json = text

    # Try to correct the json: each line should have four quotes.

    try:
        result = json.loads(potential_json)
        return result
    except json.JSONDecodeError:
        pass

    try:
        print("Try to fix json...")
        fixed_json = try_fix_json(potential_json)
        result = json.loads(fixed_json)
    except json.JSONDecodeError:
        print("Error decoding JSON after fixing. Giving up.")
        return None


def extract_text_from_pdf(pdf_file_path):
    """Extract text from a PDF file."""
    text = ''
    document = fitz.open(pdf_file_path)
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text


configs = yaml.load(open("configs.yaml", "r"), Loader=yaml.FullLoader)
agent = LegalCaseAgent(configs=configs)

files = [
    "Evans v. Linden Research, Inc., 763 F. Supp. 2d 735.pdf",
    # "Ebay, Inc. v. Bidder's Edge, Inc., 100 F. Supp. 2d 1058.pdf",
    # "HiQ Labs, Inc. v. LinkedIn Corp., 938 F.3d 985.pdf",
    # "Specht v. Netscape Communs. Corp., 306 F.3d 17.pdf",
]

for file in files:
    pdf_doc = extract_text_from_pdf(file)
    agent.parse_document(pdf_doc)
    agent.save_for_visualization(os.path.basename(os.path.abspath(file)).replace('.pdf', '.json'))

agent.save_state('processed.json')
agent.save_for_visualization('visualization.json')
print(agent.summary)

openai = OpenAI(
    api_key="aoJosgcl454OFvmHQ0XtZKopCAHKBm7S",
    base_url="https://api.deepinfra.com/v1/openai",
)


chat_completion = openai.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
print(chat_completion.choices[0].message.content)

chat_completion = openai.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user",
               "content": "Is this a law or a case?\nOlinick v. BMG Entertainment, 138 Cal. App. 4th 1286, 42 Cal. Rptr. 3d 268, 274 (Cal. Ct. App. 2006)"
               }],
    max_tokens=8192
)
print(chat_completion.choices[0].message.content)
