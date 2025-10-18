from sqlalchemy.orm import Session
from app.models.model import Model, ModelStage
from datetime import datetime

class ModelCRUD:
    def create_model(self, db: Session, model_data):
        db_model = Model(
            name=model_data["name"],
            description=model_data.get("description"),
            version=model_data.get("version", "1.0.0")
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        return db_model
    
    def get_models(self, db: Session, skip: int = 0, limit: int = 100):
        return db.query(Model).offset(skip).limit(limit).all()
    
    def get_model(self, db: Session, model_id: int):
        return db.query(Model).filter(Model.id == model_id).first()
    
    def update_model_stage(self, db: Session, model_id: int, stage: ModelStage):
        model = self.get_model(db, model_id)
        if model:
            model.stage = stage
            model.updated_at = datetime.now()
            db.commit()
            db.refresh(model)
        return model

# Create instance
model_crud = ModelCRUD()