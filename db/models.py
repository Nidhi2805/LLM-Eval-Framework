"""
Database schema for the RAG Eval Platform.

Design notes:
- SQLite now, Postgres-ready later: we use SQLAlchemy so the engine URL is the only
  thing that changes when migrating. No raw SQL tied to one dialect.
- `targets` is the extensibility seam: a target is "the thing being evaluated".
  type='builtin_rag' runs the user's docs through our pipeline; a future
  type='http_endpoint' will call the user's deployed API. `config_json` holds
  type-specific settings so new target types need no schema change.
- `runs` stores the full result JSON (source of truth, for drill-down) AND the
  summary metrics as columns (denormalized for fast trend queries).
"""

from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


def _now():
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=_now)

    documents = relationship("Document", back_populates="project",
                             cascade="all, delete-orphan")
    targets = relationship("Target", back_populates="project",
                           cascade="all, delete-orphan")
    test_sets = relationship("TestSet", back_populates="project",
                             cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="project",
                        cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    external_id = Column(String(100))   # e.g. "doc_001"; stable id used in retrieval
    title = Column(String(300))
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=_now)

    project = relationship("Project", back_populates="documents")


class Target(Base):
    __tablename__ = "targets"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(200), nullable=False)
    type = Column(String(50), default="builtin_rag")  # seam: builtin_rag | http_endpoint
    config_json = Column(Text, default="{}")          # type-specific settings
    created_at = Column(DateTime, default=_now)

    project = relationship("Project", back_populates="targets")


class TestSet(Base):
    __tablename__ = "test_sets"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(200), nullable=False)
    source = Column(String(50), default="manual")     # manual | generated
    created_at = Column(DateTime, default=_now)

    project = relationship("Project", back_populates="test_sets")
    cases = relationship("TestCase", back_populates="test_set",
                         cascade="all, delete-orphan")


class TestCase(Base):
    __tablename__ = "test_cases"
    id = Column(Integer, primary_key=True)
    test_set_id = Column(Integer, ForeignKey("test_sets.id"), nullable=False)
    external_id = Column(String(100))                 # e.g. "q_001"
    question = Column(Text, nullable=False)
    expected_answer = Column(Text, default="")
    expected_doc_ids_json = Column(Text, default="[]")  # JSON list of doc external_ids
    category = Column(String(50), default="answerable")  # answerable|unanswerable|trap

    test_set = relationship("TestSet", back_populates="cases")


class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    target_id = Column(Integer, ForeignKey("targets.id"))
    test_set_id = Column(Integer, ForeignKey("test_sets.id"))
    label = Column(String(200), default="")
    created_at = Column(DateTime, default=_now)

    # Denormalized summary metrics -> fast trend queries without parsing JSON
    faithfulness_avg = Column(Float)
    correctness_avg = Column(Float)
    refusal_accuracy = Column(Float)
    retrieval_hit = Column(Float)
    retrieval_mrr = Column(Float)

    result_json = Column(Text)   # full result: source of truth for drill-down

    project = relationship("Project", back_populates="runs")

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    status = Column(String(20), default="pending")   # pending|running|done|failed
    kind = Column(String(50), default="evaluation")
    params_json = Column(Text, default="{}")          # what was requested
    run_id = Column(Integer, ForeignKey("runs.id"))   # set when done
    error = Column(Text)                              # set when failed
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

# --------------------------------------------------------------- engine/session
DB_URL = "sqlite:///eval_platform.db"
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all tables. Safe to call repeatedly."""
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()