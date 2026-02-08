"""
Health check router.
"""

from flask import Blueprint, jsonify

health_bp = Blueprint('health', __name__)


def init_health_router(model_service, anatomy_service):
    """
    Initialize health router with services.
    
    Args:
        model_service: ModelService instance
        anatomy_service: AnatomyService instance
    """
    
    @health_bp.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_service.is_loaded(),
            'brain_data_loaded': anatomy_service.is_loaded(),
            'device': model_service.get_device()
        })
    
    return health_bp
