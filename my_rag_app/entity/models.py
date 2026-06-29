"""
SQLAlchemy ORM models for the RAG pipeline's 3-table schema:
  - Email:    raw + cleaned email record (1 row per scraped email)
  - Metadata: derived/extracted fields (1:1 with Email)
  - Chunk:    embedding-ready text unit (1:1 with Email, per "1 email = 1 chunk")
"""

from datetime import datetime

from sqlalchemy import ForeignKey, ARRAY, Text, Boolean, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from my_rag_app.constants import EMAILS_TABLE, METADATA_TABLE, CHUNKS_TABLE


class Base(DeclarativeBase):
    pass

# emails — raw + cleaned = combined
class Email(Base):
    __tablename__ = EMAILS_TABLE

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # Message-ID

    subject: Mapped[str] = mapped_column(Text, nullable=False, default="")
    body_raw: Mapped[str] = mapped_column(Text, nullable=False, default="")
    body_clean: Mapped[str | None] = mapped_column(Text, nullable=True)

    sender_email: Mapped[str] = mapped_column(Text, nullable=False, default="")
    recipient_emails: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    recipient_names: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)

    date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    thread_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    reply_to: Mapped[str | None] = mapped_column(Text, nullable=True)
    thread_match_method: Mapped[str] = mapped_column(Text, nullable=False, default="original")

    is_system_email: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    cleaned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    metadata_record: Mapped["Metadata | None"] = relationship(
        back_populates="email", uselist=False, cascade="all, delete-orphan"
    )
    chunk: Mapped["Chunk | None"] = relationship(
        back_populates="email", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Email id={self.id!r} subject={self.subject[:40]!r}>"

# metadata — derived/extracted fields, 1:1 with emails
class Metadata(Base):
    __tablename__ = METADATA_TABLE

    email_id: Mapped[str] = mapped_column(ForeignKey(f"{EMAILS_TABLE}.id"), primary_key=True)

    sender_name: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sender_company: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sender_designation: Mapped[str] = mapped_column(Text, nullable=False, default="")

    recipient_names: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)

    greeting_name: Mapped[str] = mapped_column(Text, nullable=False, default="")  # e.g. "Mr Trung" from "Dear Mr Trung,"

    extracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    email: Mapped["Email"] = relationship(back_populates="metadata_record")

    def __repr__(self) -> str:
        return f"<Metadata email_id={self.email_id!r} sender_name={self.sender_name!r}>"

# chunks — embedding-ready text unit, 1:1 with emails ("1 email = 1 chunk")
class Chunk(Base):
    __tablename__ = CHUNKS_TABLE

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)  # sha256 of email_id

    email_id: Mapped[str] = mapped_column(ForeignKey(f"{EMAILS_TABLE}.id"), nullable=False, unique=True)
    thread_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    email: Mapped["Email"] = relationship(back_populates="chunk")

    def __repr__(self) -> str:
        return f"<Chunk chunk_id={self.chunk_id!r} email_id={self.email_id!r}>"