"""Allow running as: python -m meta_eval"""
from .run import main
import asyncio

asyncio.run(main())
