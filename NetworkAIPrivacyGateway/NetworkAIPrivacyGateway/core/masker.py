import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple


class MaskerError(Exception):
    pass


@dataclass
class TokenMapping:
    original_value: str
    token: str


class BaseTokenizer(ABC):
    @abstractmethod
    def find_matches(self, text: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def format_token(self, index: int) -> str:
        raise NotImplementedError


class IPv4Tokenizer(BaseTokenizer):
    IPV4_PATTERN: Pattern[str] = re.compile(
        r"\b(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}\b"
    )

    def find_matches(self, text: str) -> List[str]:
        return list({match.group(0) for match in self.IPV4_PATTERN.finditer(text)})

    def format_token(self, index: int) -> str:
        return f"IP{index}"


@dataclass
class MaskingEngine:
    tokenizers: List[BaseTokenizer] = field(default_factory=lambda: [IPv4Tokenizer()])
    mapping: Dict[str, str] = field(default_factory=dict)
    reverse_mapping: Dict[str, str] = field(default_factory=dict)

    def mask(self, text: str) -> str:
        masked_text = text
        for tokenizer in self.tokenizers:
            matches = tokenizer.find_matches(masked_text)
            for match in matches:
                if match not in self.mapping:
                    token = tokenizer.format_token(len(self.mapping) + 1)
                    self.mapping[match] = token
                    self.reverse_mapping[token] = match
                masked_text = re.sub(rf"\b{re.escape(match)}\b", self.mapping[match], masked_text)
        return masked_text

    def unmask(self, text: str) -> str:
        restored_text = text
        for token, original in self.reverse_mapping.items():
            restored_text = restored_text.replace(token, original)
        return restored_text

    def get_mapping_table(self) -> List[TokenMapping]:
        return [TokenMapping(original_value=key, token=value) for key, value in self.mapping.items()]
