# ============= 5. Entity Extractor =============
from typing import List, Dict

from services.models import nlp


class EntityExtractor:
    """Извлечение сущностей из текста с помощью spaCy"""

    def __init__(self):
        self.nlp = nlp

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Извлечение именованных сущностей из текста

        Returns:
            List[Dict]: [{name, type, start, end}, ...]
        """
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                'name': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        return entities
