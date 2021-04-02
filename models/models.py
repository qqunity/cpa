from dataclasses import dataclass


@dataclass
class Article:
    id: int
    url: str
    title: str
    codex_type: str
    point_id: int = -1
    paragraph_id: int = -1
    chapter_id: int = -1


@dataclass
class Point:
    id: int
    url: str
    title: str
    paragraph_id: int = -1


@dataclass
class Paragraph:
    id: int
    url: str
    title: str
    chapter_id: int = -1


@dataclass
class Chapter:
    id: int
    url: str
    title: str
    subsection_id: int = -1


@dataclass
class Subsection:
    id: int
    url: str
    title: str
    section_id: int = -1


@dataclass
class Section:
    id: int
    url: str
    title: str
    codex_id: int = -1


@dataclass
class Codex:
    id: int
    url: str
    title: str
