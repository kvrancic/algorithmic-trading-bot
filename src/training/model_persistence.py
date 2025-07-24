"""
Model Persistence and Loading System

Comprehensive system for saving, loading, and managing trained models.
Includes versioning, metadata tracking, and model registry.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
import joblib
import shutil
import hashlib
import structlog
from abc import ABC, abstractmethod
from collections import defaultdict

from ..models.base import BaseModel

logger = structlog.get_logger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a saved model"""
    
    model_id: str
    model_name: str
    model_type: str
    version: str
    created_timestamp: datetime
    last_updated: datetime
    
    # Training information
    training_data_hash: str
    training_samples: int
    training_duration: float
    training_config: Dict[str, Any]
    
    # Performance metrics
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    
    # Model artifacts paths
    model_path: Path
    config_path: Path
    metadata_path: Path
    
    # Dependencies and environment
    python_version: str
    dependencies: Dict[str, str]
    environment_hash: str
    
    # Additional metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'created_timestamp': self.created_timestamp.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'training_data_hash': self.training_data_hash,
            'training_samples': self.training_samples,
            'training_duration': self.training_duration,
            'training_config': self.training_config,
            'validation_metrics': self.validation_metrics,
            'test_metrics': self.test_metrics,
            'model_path': str(self.model_path),
            'config_path': str(self.config_path),
            'metadata_path': str(self.metadata_path),
            'python_version': self.python_version,
            'dependencies': self.dependencies,
            'environment_hash': self.environment_hash,
            'description': self.description,
            'tags': self.tags,
            'author': self.author
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        return cls(
            model_id=data['model_id'],
            model_name=data['model_name'],
            model_type=data['model_type'],
            version=data['version'],
            created_timestamp=datetime.fromisoformat(data['created_timestamp']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            training_data_hash=data['training_data_hash'],
            training_samples=data['training_samples'],
            training_duration=data['training_duration'],
            training_config=data['training_config'],
            validation_metrics=data['validation_metrics'],
            test_metrics=data['test_metrics'],
            model_path=Path(data['model_path']),
            config_path=Path(data['config_path']),
            metadata_path=Path(data['metadata_path']),
            python_version=data['python_version'],
            dependencies=data['dependencies'],
            environment_hash=data['environment_hash'],
            description=data.get('description', ''),
            tags=data.get('tags', []),
            author=data.get('author', '')
        )


@dataclass
class PersistenceConfig:
    """Configuration for model persistence"""
    
    # Storage configuration
    model_registry_path: Path = field(default_factory=lambda: Path("model_registry"))
    compression: bool = True
    backup_models: bool = True
    max_versions_per_model: int = 5
    
    # Security
    encrypt_models: bool = False
    encryption_key: Optional[str] = None
    
    # Validation
    validate_on_load: bool = True
    check_dependencies: bool = True
    
    # Cleanup
    auto_cleanup: bool = True
    cleanup_threshold_days: int = 30
    min_models_to_keep: int = 2
    
    # Model registry
    use_model_registry: bool = True
    registry_file: str = "model_registry.json"
    
    def __post_init__(self):
        self.model_registry_path = Path(self.model_registry_path)
        self.model_registry_path.mkdir(exist_ok=True, parents=True)


class ModelRegistry:
    """Registry for managing model metadata and versions"""
    
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_file = registry_path / "registry.json"
        self.models: Dict[str, Dict[str, ModelMetadata]] = defaultdict(dict)
        self.load_registry()
    
    def load_registry(self):
        """Load registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                for model_name, versions in data.items():
                    for version, metadata_dict in versions.items():
                        self.models[model_name][version] = ModelMetadata.from_dict(metadata_dict)
                
                logger.info("Model registry loaded", 
                           n_models=len(self.models),
                           total_versions=sum(len(versions) for versions in self.models.values()))
                
            except Exception as e:
                logger.error("Failed to load model registry", error=str(e))
                self.models = defaultdict(dict)
    
    def save_registry(self):
        """Save registry to file"""
        try:
            # Convert to serializable format
            data = {}
            for model_name, versions in self.models.items():
                data[model_name] = {}
                for version, metadata in versions.items():
                    data[model_name][version] = metadata.to_dict()
            
            # Atomic write
            temp_file = self.registry_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.registry_file)
            
            logger.info("Model registry saved")
            
        except Exception as e:
            logger.error("Failed to save model registry", error=str(e))
    
    def register_model(self, metadata: ModelMetadata):
        """Register a new model version"""
        self.models[metadata.model_name][metadata.version] = metadata
        self.save_registry()
        
        logger.info("Model registered", 
                   model_name=metadata.model_name, 
                   version=metadata.version)
    
    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """Get model metadata"""
        if model_name not in self.models:
            return None
        
        if version is None:
            # Get latest version
            versions = list(self.models[model_name].keys())
            if not versions:
                return None
            version = max(versions)  # Assuming version strings are comparable
        
        return self.models[model_name].get(version)
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all models and their versions"""
        return {name: list(versions.keys()) for name, versions in self.models.items()}
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """Delete a specific model version"""
        if model_name in self.models and version in self.models[model_name]:
            del self.models[model_name][version]
            
            # Remove model entirely if no versions left
            if not self.models[model_name]:
                del self.models[model_name]
            
            self.save_registry()
            return True
        
        return False
    
    def cleanup_old_versions(self, model_name: str, max_versions: int):
        """Keep only the latest N versions of a model"""
        if model_name not in self.models:
            return
        
        versions = list(self.models[model_name].keys())
        if len(versions) <= max_versions:
            return
        
        # Sort versions (assuming semantic versioning or timestamp-based)
        versions.sort(reverse=True)  # Keep latest first
        
        # Remove old versions
        for version in versions[max_versions:]:
            metadata = self.models[model_name][version]
            
            # Delete model files
            try:
                if metadata.model_path.exists():
                    if metadata.model_path.is_dir():
                        shutil.rmtree(metadata.model_path)
                    else:
                        metadata.model_path.unlink()
                
                if metadata.config_path.exists():
                    metadata.config_path.unlink()
                
                if metadata.metadata_path.exists():
                    metadata.metadata_path.unlink()
                
            except Exception as e:
                logger.error(f"Failed to delete files for {model_name} v{version}", error=str(e))
            
            # Remove from registry
            del self.models[model_name][version]
        
        self.save_registry()
        
        logger.info("Cleaned up old model versions",
                   model_name=model_name,
                   versions_removed=len(versions) - max_versions)


class ModelPersistence:
    """Main class for model persistence and loading"""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.registry = ModelRegistry(config.model_registry_path) if config.use_model_registry else None
    
    def save_model(
        self,
        model: BaseModel,
        model_name: str,
        training_data_hash: str,
        training_samples: int,
        training_duration: float,
        validation_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        description: str = "",
        tags: List[str] = None,
        author: str = ""
    ) -> ModelMetadata:
        """Save a trained model with full metadata"""
        
        logger.info(f"Saving model: {model_name}")
        
        # Generate model ID and version
        model_id = self._generate_model_id(model_name, training_data_hash)
        version = self._generate_version(model_name)
        
        # Create model directory
        model_dir = self.config.model_registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using its native save method
        model_path = model_dir / "model"
        saved_path = model.save(model_path)
        
        # Save configuration
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(model.config.to_dict(), f, indent=2)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model.__class__.__name__,
            version=version,
            created_timestamp=datetime.now(),
            last_updated=datetime.now(),
            training_data_hash=training_data_hash,
            training_samples=training_samples,
            training_duration=training_duration,
            training_config=model.config.to_dict(),
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            model_path=saved_path,
            config_path=config_path,
            metadata_path=model_dir / "metadata.json",
            python_version=self._get_python_version(),
            dependencies=self._get_dependencies(),
            environment_hash=self._get_environment_hash(),
            description=description,
            tags=tags or [],
            author=author
        )
        
        # Save metadata
        with open(metadata.metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Register model
        if self.registry:
            self.registry.register_model(metadata)
        
        # Cleanup old versions if configured
        if self.config.auto_cleanup and self.registry:
            self.registry.cleanup_old_versions(
                model_name, 
                self.config.max_versions_per_model
            )
        
        logger.info(f"Model saved successfully", 
                   model_id=model_id, 
                   version=version,
                   path=str(model_dir))
        
        return metadata
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        model_class: Optional[Type[BaseModel]] = None
    ) -> Tuple[BaseModel, ModelMetadata]:
        """Load a saved model"""
        
        logger.info(f"Loading model: {model_name}", version=version)
        
        # Get metadata
        if self.registry:
            metadata = self.registry.get_model_metadata(model_name, version)
        else:
            # Fallback: look for metadata file directly
            if version:
                metadata_path = self.config.model_registry_path / model_name / version / "metadata.json"
            else:
                # Find latest version
                model_dir = self.config.model_registry_path / model_name
                if not model_dir.exists():
                    raise ValueError(f"Model {model_name} not found")
                
                versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
                if not versions:
                    raise ValueError(f"No versions found for model {model_name}")
                
                version = max(versions)  # Get latest
                metadata_path = model_dir / version / "metadata.json"
            
            if not metadata_path.exists():
                raise ValueError(f"Metadata not found for {model_name} v{version}")
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata.from_dict(metadata_dict)
        
        if metadata is None:
            raise ValueError(f"Model {model_name} version {version} not found")
        
        # Validate dependencies if configured
        if self.config.check_dependencies:
            self._validate_dependencies(metadata)
        
        # Determine model class
        if model_class is None:
            model_class = self._get_model_class(metadata.model_type)
        
        # Load model
        try:
            model = model_class.load(metadata.model_path)
            
            # Validate model if configured
            if self.config.validate_on_load:
                self._validate_loaded_model(model, metadata)
            
            logger.info(f"Model loaded successfully", 
                       model_name=model_name, 
                       version=metadata.version)
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}", error=str(e))
            raise
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models and versions"""
        if self.registry:
            return self.registry.list_models()
        else:
            # Scan directory structure
            models = {}
            registry_path = self.config.model_registry_path
            
            if not registry_path.exists():
                return models
            
            for model_dir in registry_path.iterdir():
                if model_dir.is_dir():
                    versions = [v.name for v in model_dir.iterdir() if v.is_dir()]
                    if versions:
                        models[model_dir.name] = sorted(versions, reverse=True)
            
            return models
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Delete a model version or all versions"""
        
        if version:
            # Delete specific version
            if self.registry:
                success = self.registry.delete_model_version(model_name, version)
            else:
                model_dir = self.config.model_registry_path / model_name / version
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    success = True
                else:
                    success = False
        else:
            # Delete all versions
            model_dir = self.config.model_registry_path / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
                
                # Remove from registry
                if self.registry and model_name in self.registry.models:
                    del self.registry.models[model_name]
                    self.registry.save_registry()
                
                success = True
            else:
                success = False
        
        if success:
            logger.info("Model deleted", model_name=model_name, version=version)
        else:
            logger.warning("Model not found for deletion", model_name=model_name, version=version)
        
        return success
    
    def export_model(
        self,
        model_name: str,
        version: Optional[str],
        export_path: Path,
        include_dependencies: bool = True
    ):
        """Export model to a portable format"""
        
        logger.info(f"Exporting model: {model_name}", version=version, export_path=str(export_path))
        
        # Load model metadata
        metadata = self.registry.get_model_metadata(model_name, version) if self.registry else None
        if metadata is None:
            raise ValueError(f"Model {model_name} version {version} not found")
        
        # Create export directory
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_export_dir = export_path / "model"
        if metadata.model_path.is_dir():
            shutil.copytree(metadata.model_path, model_export_dir)
        else:
            shutil.copy2(metadata.model_path, model_export_dir)
        
        # Copy config
        shutil.copy2(metadata.config_path, export_path / "config.json")
        
        # Copy metadata
        shutil.copy2(metadata.metadata_path, export_path / "metadata.json")
        
        # Create requirements.txt if requested
        if include_dependencies:
            requirements_path = export_path / "requirements.txt"
            with open(requirements_path, 'w') as f:
                for package, version in metadata.dependencies.items():
                    f.write(f"{package}=={version}\n")
        
        # Create export manifest
        manifest = {
            'model_name': model_name,
            'version': metadata.version,
            'export_timestamp': datetime.now().isoformat(),
            'exported_by': 'ModelPersistence',
            'files': [
                'model',
                'config.json',
                'metadata.json'
            ]
        }
        
        if include_dependencies:
            manifest['files'].append('requirements.txt')
        
        with open(export_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info("Model exported successfully", export_path=str(export_path))
    
    def import_model(self, import_path: Path, model_name: Optional[str] = None) -> ModelMetadata:
        """Import a model from exported format"""
        
        logger.info("Importing model", import_path=str(import_path))
        
        import_path = Path(import_path)
        
        # Load manifest
        manifest_path = import_path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Invalid export: manifest.json not found")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Load metadata
        metadata_path = import_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Override model name if provided
        if model_name:
            metadata_dict['model_name'] = model_name
        
        # Generate new version
        original_name = metadata_dict['model_name']
        new_version = self._generate_version(original_name)
        metadata_dict['version'] = new_version
        
        # Create new model directory
        model_dir = self.config.model_registry_path / original_name / new_version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        shutil.copytree(import_path / "model", model_dir / "model")
        shutil.copy2(import_path / "config.json", model_dir / "config.json")
        
        # Update paths in metadata
        metadata_dict['model_path'] = str(model_dir / "model")
        metadata_dict['config_path'] = str(model_dir / "config.json")
        metadata_dict['metadata_path'] = str(model_dir / "metadata.json")
        metadata_dict['last_updated'] = datetime.now().isoformat()
        
        # Save metadata
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Create metadata object
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        # Register model
        if self.registry:
            self.registry.register_model(metadata)
        
        logger.info("Model imported successfully", 
                   model_name=original_name, 
                   version=new_version)
        
        return metadata
    
    def _generate_model_id(self, model_name: str, training_data_hash: str) -> str:
        """Generate unique model ID"""
        content = f"{model_name}_{training_data_hash}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_version(self, model_name: str) -> str:
        """Generate version string"""
        # Simple timestamp-based versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add counter if version already exists
        counter = 1
        version = timestamp
        
        if self.registry:
            while version in self.registry.models.get(model_name, {}):
                version = f"{timestamp}_{counter:02d}"
                counter += 1
        
        return version
    
    def _get_python_version(self) -> str:
        """Get current Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current package versions"""
        import pkg_resources
        
        dependencies = {}
        
        # Key packages for trading models
        key_packages = [
            'numpy', 'pandas', 'scikit-learn', 'torch', 'transformers',
            'xgboost', 'matplotlib', 'seaborn', 'structlog'
        ]
        
        for package in key_packages:
            try:
                version = pkg_resources.get_distribution(package).version
                dependencies[package] = version
            except pkg_resources.DistributionNotFound:
                pass
        
        return dependencies
    
    def _get_environment_hash(self) -> str:
        """Generate hash of current environment"""
        import sys
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'dependencies': self._get_dependencies()
        }
        
        env_str = json.dumps(env_info, sort_keys=True)
        return hashlib.md5(env_str.encode()).hexdigest()
    
    def _validate_dependencies(self, metadata: ModelMetadata):
        """Validate that required dependencies are available"""
        
        current_deps = self._get_dependencies()
        missing_deps = []
        version_mismatches = []
        
        for package, required_version in metadata.dependencies.items():
            if package not in current_deps:
                missing_deps.append(package)
            elif current_deps[package] != required_version:
                version_mismatches.append(f"{package}: required {required_version}, found {current_deps[package]}")
        
        if missing_deps:
            logger.warning("Missing dependencies", packages=missing_deps)
        
        if version_mismatches:
            logger.warning("Version mismatches", mismatches=version_mismatches)
    
    def _get_model_class(self, model_type: str) -> Type[BaseModel]:
        """Get model class from model type string"""
        
        # Import model classes
        from ..models import (
            PriceLSTM, ChartPatternCNN, MarketRegimeXGBoost, 
            FinBERT, StackedEnsemble
        )
        
        model_classes = {
            'PriceLSTM': PriceLSTM,
            'ChartPatternCNN': ChartPatternCNN,
            'MarketRegimeXGBoost': MarketRegimeXGBoost,
            'FinBERT': FinBERT,
            'StackedEnsemble': StackedEnsemble
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_classes[model_type]
    
    def _validate_loaded_model(self, model: BaseModel, metadata: ModelMetadata):
        """Validate that loaded model is working correctly"""
        
        # Basic validation - check that model has required attributes
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded model does not have predict method")
        
        if not hasattr(model, 'is_trained') or not model.is_trained:
            raise ValueError("Loaded model is not trained")
        
        # Additional model-specific validations could be added here
        logger.info("Model validation passed")
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        
        if self.registry:
            metadata = self.registry.get_model_metadata(model_name, version)
            if metadata:
                return metadata.to_dict()
        
        return None
    
    def cleanup_old_models(self, days_threshold: int = None):
        """Clean up old models based on age"""
        
        threshold = days_threshold or self.config.cleanup_threshold_days
        cutoff_date = datetime.now() - timedelta(days=threshold)
        
        if not self.registry:
            logger.warning("Cannot cleanup models without registry")
            return
        
        cleaned_models = []
        
        for model_name, versions in list(self.registry.models.items()):
            # Keep track of versions to delete
            versions_to_delete = []
            
            # Sort versions by creation date
            version_items = list(versions.items())
            version_items.sort(key=lambda x: x[1].created_timestamp, reverse=True)
            
            # Keep minimum number of recent models
            recent_versions = version_items[:self.config.min_models_to_keep]
            old_versions = version_items[self.config.min_models_to_keep:]
            
            # Delete old versions that exceed the threshold
            for version, metadata in old_versions:
                if metadata.created_timestamp < cutoff_date:
                    versions_to_delete.append(version)
            
            # Delete identified versions
            for version in versions_to_delete:
                if self.registry.delete_model_version(model_name, version):
                    cleaned_models.append(f"{model_name} v{version}")
        
        if cleaned_models:
            logger.info("Cleaned up old models", models=cleaned_models)
        else:
            logger.info("No old models to clean up")