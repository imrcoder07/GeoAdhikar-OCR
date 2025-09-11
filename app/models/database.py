"""
Database models for FRA OCR system
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.config import Config

Base = declarative_base()

class OCRDocument(Base):
    """Model for storing OCR document records"""
    __tablename__ = 'ocr_documents'
    
    id = Column(Integer, primary_key=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500))
    upload_date = Column(DateTime, default=datetime.utcnow)
    process_date = Column(DateTime)
    
    # Extracted fields
    patta_id = Column(String(100))
    holder_name = Column(String(255))
    relative_name = Column(String(255))
    relation_type = Column(String(50))
    village = Column(String(255))
    district = Column(String(100))
    state = Column(String(100))
    area_hectares = Column(Float)
    issue_date = Column(DateTime)
    claim_type = Column(String(20))  # IFR, CR, CFR
    claim_status = Column(String(50))  # granted, pending, rejected
    
    # OCR metadata
    raw_ocr_text = Column(Text)
    structured_data = Column(JSON)  # Store complete structured response
    language_detected = Column(String(100))
    ocr_confidence = Column(Float)
    quality_score = Column(Float)
    
    # Processing metadata
    processing_status = Column(String(50), default='pending')  # pending, processing, completed, failed
    error_message = Column(Text)
    processing_time_ms = Column(Integer)
    
    # Validation flags
    needs_verification = Column(Boolean, default=False)
    verification_reasons = Column(JSON)
    verified_by = Column(String(100))
    verified_date = Column(DateTime)
    
    # Coordinates if available
    latitude = Column(Float)
    longitude = Column(Float)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'file_name': self.file_name,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'patta_id': self.patta_id,
            'holder_name': self.holder_name,
            'relative_name': self.relative_name,
            'village': self.village,
            'district': self.district,
            'state': self.state,
            'area_hectares': self.area_hectares,
            'issue_date': self.issue_date.isoformat() if self.issue_date else None,
            'claim_type': self.claim_type,
            'claim_status': self.claim_status,
            'ocr_confidence': self.ocr_confidence,
            'processing_status': self.processing_status,
            'needs_verification': self.needs_verification
        }

class ProcessingLog(Base):
    """Model for storing processing logs"""
    __tablename__ = 'processing_logs'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String(100))
    details = Column(Text)
    user = Column(String(100))

# Database setup
engine = create_engine(Config.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
