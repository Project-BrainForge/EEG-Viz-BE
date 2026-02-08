"""
Brain data router for anatomy information.
"""

from flask import Blueprint, jsonify

brain_bp = Blueprint('brain', __name__)


def init_brain_router(anatomy_service, config):
    """
    Initialize brain router with services.
    
    Args:
        anatomy_service: AnatomyService instance
        config: Configuration dictionary with N_SOURCES
    """
    N_SOURCES = config['N_SOURCES']
    
    @brain_bp.route('/api/brain-data', methods=['GET'])
    def get_brain_data():
        """Get brain mesh data without predictions."""
        try:
            response_data = anatomy_service.get_brain_data_dict(N_SOURCES)
            return jsonify(response_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return brain_bp
