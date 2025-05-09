import debugg_ai_sdk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


company_key = 'a9179c1c-94fc-4c9b-9bcf-3a442407426e'
project_key = 'acefe594-49d0-4d43-ab4b-b38515fef8fe'


debugg_ai_sdk.init(
    dsn=f"https://debuggai-backend.ngrok.app/api/v1/ingest/{company_key}/{project_key}/",
    environment="local",
    host_name="qosha-local",
)

class TestClass:
    def __init__(self):
        
        self.test_var = "test"
        
    def divide_by_zero(self):
        return 1 / 0

def test_function():
    # Fundtion that pretends to do something
    logger.info("Doing something")
    
    test_class = TestClass()
    test_class.divide_by_zero()
    

def main():
    logger.info("Hello, world!")

    test_function()
if __name__ == "__main__":
    main()