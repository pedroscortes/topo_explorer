import asyncio
import logging
from topo_explorer.visualization.test_data_generator import TestDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection():
    try:
        logger.info("Starting connection test...")
        generator = TestDataGenerator()
        
        # Test manifold data generation
        logger.info("Testing manifold data generation...")
        manifold_data = generator.generate_manifold_data()
        logger.info(f"Generated manifold data: {manifold_data}")
        
        # Test trajectory generation
        logger.info("Testing trajectory generation...")
        trajectory_data = generator.generate_trajectory_point()
        logger.info(f"Generated trajectory data: {trajectory_data}")
        
        # Test metrics generation
        logger.info("Testing metrics generation...")
        metrics_data = generator.generate_metrics()
        logger.info(f"Generated metrics data: {metrics_data}")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_connection())