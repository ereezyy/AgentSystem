"""
Browser Module
-------------
Provides browser automation capabilities for the agent
"""

import os
import time
import base64
import asyncio
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field

# Local imports
from AgentSystem.utils.logger import get_logger
from AgentSystem.utils.env_loader import get_env

# Get module logger
logger = get_logger("modules.browser")

# Try to import Playwright
try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    logger.warning("Playwright package not installed. Browser automation will not be available.")
    PLAYWRIGHT_AVAILABLE = False


@dataclass
class BrowserAction:
    """Represents a browser action"""
    action_type: str  # 'goto', 'click', 'type', 'screenshot', etc.
    params: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class BrowserModule:
    """
    Browser automation module
    
    Allows the agent to control a web browser for various tasks
    """
    
    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium",
        user_data_dir: Optional[str] = None,
        viewport_size: Tuple[int, int] = (1280, 720),
        slow_mo: int = 50,
        timeout: int = 30000,
        memory_manager = None
    ):
        """
        Initialize the browser module
        
        Args:
            headless: Whether to run in headless mode
            browser_type: Type of browser ('chromium', 'firefox', or 'webkit')
            user_data_dir: Directory for user data (for persistent sessions)
            viewport_size: Size of the viewport (width, height)
            slow_mo: Slow down actions by this amount (ms)
            timeout: Default timeout for actions (ms)
            memory_manager: Memory manager for storing screenshots and results
        """
        self.headless = headless
        self.browser_type = browser_type
        self.user_data_dir = user_data_dir
        self.viewport_size = viewport_size
        self.slow_mo = slow_mo
        self.timeout = timeout
        self.memory_manager = memory_manager
        
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._action_history: List[BrowserAction] = []
        
        self._event_loop = None
        
        logger.debug(f"Browser module initialized with {browser_type} browser")
    
    async def _initialize_browser(self) -> None:
        """Initialize the browser instance"""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright is not available. Cannot initialize browser.")
        
        try:
            self._playwright = await async_playwright().start()
            
            # Get the browser instance
            browser_class = getattr(self._playwright, self.browser_type)
            
            launch_options = {
                "headless": self.headless,
                "slow_mo": self.slow_mo
            }
            
            # If we have a user data dir, add it to the launch options
            if self.user_data_dir:
                os.makedirs(self.user_data_dir, exist_ok=True)
                if self.browser_type == "chromium":
                    launch_options["user_data_dir"] = self.user_data_dir
            
            self._browser = await browser_class.launch(**launch_options)
            
            # Create a context
            self._context = await self._browser.new_context(
                viewport={"width": self.viewport_size[0], "height": self.viewport_size[1]},
                accept_downloads=True
            )
            
            # Set default timeout
            self._context.set_default_timeout(self.timeout)
            
            # Create a page
            self._page = await self._context.new_page()
            
            # Listen for console messages
            self._page.on("console", lambda msg: logger.debug(f"Browser console: {msg.text}"))
            
            logger.info(f"{self.browser_type.capitalize()} browser initialized")
            
        except Exception as e:
            logger.error(f"Error initializing browser: {e}")
            await self._cleanup()
            raise
    
    async def _cleanup(self) -> None:
        """Clean up resources"""
        try:
            if self._page:
                await self._page.close()
                self._page = None
            
            if self._context:
                await self._context.close()
                self._context = None
            
            if self._browser:
                await self._browser.close()
                self._browser = None
            
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
                
            logger.debug("Browser resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up browser resources: {e}")
    
    def _add_action(self, action_type: str, params: Dict[str, Any]) -> BrowserAction:
        """
        Add an action to the history
        
        Args:
            action_type: Type of action
            params: Action parameters
            
        Returns:
            Browser action
        """
        action = BrowserAction(action_type=action_type, params=params)
        self._action_history.append(action)
        
        # Store in memory if we have a memory manager
        if self.memory_manager:
            self.memory_manager.add(
                content={
                    "action_type": action_type,
                    "params": params,
                    "timestamp": action.timestamp
                },
                memory_type="browser_action",
                importance=0.6,
                metadata={
                    "module": "browser",
                    "action_type": action_type
                }
            )
        
        return action
    
    def _update_action_result(self, action: BrowserAction, result: Any = None, error: Optional[str] = None) -> None:
        """
        Update an action with its result
        
        Args:
            action: Browser action
            result: Action result
            error: Error message if any
        """
        action.result = result
        action.error = error
        
        # Update in memory if we have a memory manager
        if self.memory_manager and action.error:
            self.memory_manager.add(
                content={
                    "action_type": action.action_type,
                    "params": action.params,
                    "error": action.error,
                    "timestamp": action.timestamp
                },
                memory_type="browser_error",
                importance=0.8,
                metadata={
                    "module": "browser",
                    "action_type": action.action_type,
                    "has_error": True
                }
            )
    
    async def _get_screenshot(self, full_page: bool = False) -> str:
        """
        Take a screenshot of the current page
        
        Args:
            full_page: Whether to capture the full page
            
        Returns:
            Base64-encoded screenshot
        """
        if not self._page:
            return ""
            
        try:
            # Create a temporary file for the screenshot
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            
            # Take screenshot
            await self._page.screenshot(path=tmp_path, full_page=full_page)
            
            # Read the file and encode as base64
            with open(tmp_path, "rb") as f:
                img_data = f.read()
                
            # Clean up the temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            # Encode as base64
            return base64.b64encode(img_data).decode("utf-8")
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return ""
    
    def _ensure_event_loop(self) -> asyncio.AbstractEventLoop:
        """
        Ensure we have an event loop
        
        Returns:
            Event loop
        """
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    def _run_async(self, coro):
        """
        Run a coroutine in the event loop
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Result of the coroutine
        """
        loop = self._ensure_event_loop()
        
        if loop.is_running():
            # Already running, use asyncio.run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        else:
            # Not running, use loop.run_until_complete
            return loop.run_until_complete(coro)
    
    def start(self) -> bool:
        """
        Start the browser
        
        Returns:
            Success flag
        """
        action = self._add_action("start", {})
        
        try:
            self._run_async(self._initialize_browser())
            self._update_action_result(action, result=True)
            logger.info("Browser started successfully")
            return True
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error starting browser: {error_msg}")
            self._update_action_result(action, error=error_msg)
            return False
    
    def stop(self) -> bool:
        """
        Stop the browser
        
        Returns:
            Success flag
        """
        action = self._add_action("stop", {})
        
        try:
            self._run_async(self._cleanup())
            self._update_action_result(action, result=True)
            logger.info("Browser stopped successfully")
            return True
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error stopping browser: {error_msg}")
            self._update_action_result(action, error=error_msg)
            return False
    
    def navigate(self, url: str, wait_until: str = "load") -> Dict[str, Any]:
        """
        Navigate to a URL
        
        Args:
            url: URL to navigate to
            wait_until: When to consider navigation complete
                        ('load', 'domcontentloaded', 'networkidle')
            
        Returns:
            Result dict with screenshot and status
        """
        action = self._add_action("navigate", {"url": url, "wait_until": wait_until})
        
        if not self._page:
            error_msg = "Browser not initialized"
            logger.error(error_msg)
            self._update_action_result(action, error=error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            async def _navigate():
                await self._page.goto(url, wait_until=wait_until)
                screenshot = await self._get_screenshot()
                title = await self._page.title()
                return {"screenshot": screenshot, "title": title}
            
            result = self._run_async(_navigate())
            self._update_action_result(action, result={"url": url, "title": result["title"]})
            
            logger.info(f"Navigated to {url}")
            
            return {
                "success": True,
                "screenshot": result["screenshot"],
                "title": result["title"],
                "url": url
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error navigating to {url}: {error_msg}")
            self._update_action_result(action, error=error_msg)
            
            # Try to get a screenshot anyway
            try:
                screenshot = self._run_async(self._get_screenshot())
            except:
                screenshot = ""
                
            return {
                "success": False,
                "error": error_msg,
                "screenshot": screenshot,
                "url": url
            }
    
    def click(self, selector: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Click on an element
        
        Args:
            selector: Element selector
            timeout: Timeout in milliseconds
            
        Returns:
            Result dict with screenshot and status
        """
        action = self._add_action("click", {"selector": selector, "timeout": timeout})
        
        if not self._page:
            error_msg = "Browser not initialized"
            logger.error(error_msg)
            self._update_action_result(action, error=error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            async def _click():
                click_options = {}
                if timeout:
                    click_options["timeout"] = timeout
                
                await self._page.click(selector, **click_options)
                screenshot = await self._get_screenshot()
                return {"screenshot": screenshot}
            
            result = self._run_async(_click())
            self._update_action_result(action, result={"selector": selector})
            
            logger.info(f"Clicked on element {selector}")
            
            return {
                "success": True,
                "screenshot": result["screenshot"]
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error clicking on {selector}: {error_msg}")
            self._update_action_result(action, error=error_msg)
            
            # Try to get a screenshot anyway
            try:
                screenshot = self._run_async(self._get_screenshot())
            except:
                screenshot = ""
                
            return {
                "success": False,
                "error": error_msg,
                "screenshot": screenshot
            }
    
    def type_text(self, selector: str, text: str, delay: int = 50) -> Dict[str, Any]:
        """
        Type text into an element
        
        Args:
            selector: Element selector
            text: Text to type
            delay: Delay between keystrokes in milliseconds
            
        Returns:
            Result dict with screenshot and status
        """
        action = self._add_action("type", {"selector": selector, "text": text, "delay": delay})
        
        if not self._page:
            error_msg = "Browser not initialized"
            logger.error(error_msg)
            self._update_action_result(action, error=error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            async def _type():
                await self._page.fill(selector, text, timeout=self.timeout)
                screenshot = await self._get_screenshot()
                return {"screenshot": screenshot}
            
            result = self._run_async(_type())
            self._update_action_result(action, result={"selector": selector, "text_length": len(text)})
            
            logger.info(f"Typed text into element {selector}")
            
            return {
                "success": True,
                "screenshot": result["screenshot"]
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error typing into {selector}: {error_msg}")
            self._update_action_result(action, error=error_msg)
            
            # Try to get a screenshot anyway
            try:
                screenshot = self._run_async(self._get_screenshot())
            except:
                screenshot = ""
                
            return {
                "success": False,
                "error": error_msg,
                "screenshot": screenshot
            }
    
    def screenshot(self, full_page: bool = False) -> Dict[str, Any]:
        """
        Take a screenshot
        
        Args:
            full_page: Whether to capture the full page
            
        Returns:
            Result dict with screenshot and status
        """
        action = self._add_action("screenshot", {"full_page": full_page})
        
        if not self._page:
            error_msg = "Browser not initialized"
            logger.error(error_msg)
            self._update_action_result(action, error=error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            screenshot = self._run_async(self._get_screenshot(full_page=full_page))
            self._update_action_result(action, result={"full_page": full_page})
            
            logger.info("Took screenshot")
            
            return {
                "success": True,
                "screenshot": screenshot
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error taking screenshot: {error_msg}")
            self._update_action_result(action, error=error_msg)
            
            return {
                "success": False,
                "error": error_msg
            }
    
    def extract_text(self, selector: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text from the page or a specific element
        
        Args:
            selector: Element selector (or None for the entire page)
            
        Returns:
            Result dict with text and status
        """
        action = self._add_action("extract_text", {"selector": selector})
        
        if not self._page:
            error_msg = "Browser not initialized"
            logger.error(error_msg)
            self._update_action_result(action, error=error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            async def _extract_text():
                if selector:
                    # Extract text from a specific element
                    text = await self._page.text_content(selector)
                else:
                    # Extract text from the entire page body
                    text = await self._page.text_content("body")
                
                screenshot = await self._get_screenshot()
                return {"text": text, "screenshot": screenshot}
            
            result = self._run_async(_extract_text())
            text = result["text"]
            
            # Store in memory if we have a memory manager
            if self.memory_manager:
                current_url = self._run_async(lambda: self._page.url)
                self.memory_manager.add(
                    content=text,
                    memory_type="web_content",
                    importance=0.7,
                    metadata={
                        "module": "browser",
                        "action_type": "extract_text",
                        "url": current_url,
                        "selector": selector
                    }
                )
            
            self._update_action_result(action, result={"selector": selector, "text_length": len(text)})
            
            logger.info(f"Extracted text from {selector or 'page'}")
            
            return {
                "success": True,
                "text": text,
                "screenshot": result["screenshot"]
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error extracting text from {selector or 'page'}: {error_msg}")
            self._update_action_result(action, error=error_msg)
            
            # Try to get a screenshot anyway
            try:
                screenshot = self._run_async(self._get_screenshot())
            except:
                screenshot = ""
                
            return {
                "success": False,
                "error": error_msg,
                "screenshot": screenshot
            }
    
    def evaluate_javascript(self, script: str) -> Dict[str, Any]:
        """
        Evaluate JavaScript on the page
        
        Args:
            script: JavaScript code to evaluate
            
        Returns:
            Result dict with result and status
        """
        action = self._add_action("evaluate_javascript", {"script": script})
        
        if not self._page:
            error_msg = "Browser not initialized"
            logger.error(error_msg)
            self._update_action_result(action, error=error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            async def _evaluate():
                result = await self._page.evaluate(script)
                screenshot = await self._get_screenshot()
                return {"result": result, "screenshot": screenshot}
            
            result = self._run_async(_evaluate())
            self._update_action_result(action, result={"result": result["result"]})
            
            logger.info("Evaluated JavaScript")
            
            return {
                "success": True,
                "result": result["result"],
                "screenshot": result["screenshot"]
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error evaluating JavaScript: {error_msg}")
            self._update_action_result(action, error=error_msg)
            
            # Try to get a screenshot anyway
            try:
                screenshot = self._run_async(self._get_screenshot())
            except:
                screenshot = ""
                
            return {
                "success": False,
                "error": error_msg,
                "screenshot": screenshot
            }
    
    def wait_for_selector(self, selector: str, timeout: Optional[int] = None, state: str = "visible") -> Dict[str, Any]:
        """
        Wait for an element to appear
        
        Args:
            selector: Element selector
            timeout: Timeout in milliseconds
            state: Element state to wait for ('attached', 'detached', 'visible', 'hidden')
            
        Returns:
            Result dict with screenshot and status
        """
        action = self._add_action("wait_for_selector", {"selector": selector, "timeout": timeout, "state": state})
        
        if not self._page:
            error_msg = "Browser not initialized"
            logger.error(error_msg)
            self._update_action_result(action, error=error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            async def _wait():
                wait_options = {"state": state}
                if timeout:
                    wait_options["timeout"] = timeout
                
                await self._page.wait_for_selector(selector, **wait_options)
                screenshot = await self._get_screenshot()
                return {"screenshot": screenshot}
            
            result = self._run_async(_wait())
            self._update_action_result(action, result={"selector": selector, "state": state})
            
            logger.info(f"Waited for selector {selector} to be {state}")
            
            return {
                "success": True,
                "screenshot": result["screenshot"]
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error waiting for selector {selector}: {error_msg}")
            self._update_action_result(action, error=error_msg)
            
            # Try to get a screenshot anyway
            try:
                screenshot = self._run_async(self._get_screenshot())
            except:
                screenshot = ""
                
            return {
                "success": False,
                "error": error_msg,
                "screenshot": screenshot
            }
    
    def get_current_url(self) -> Dict[str, Any]:
        """
        Get the current URL
        
        Returns:
            Result dict with URL and status
        """
        action = self._add_action("get_current_url", {})
        
        if not self._page:
            error_msg = "Browser not initialized"
            logger.error(error_msg)
            self._update_action_result(action, error=error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            async def _get_url():
                url = self._page.url
                screenshot = await self._get_screenshot()
                return {"url": url, "screenshot": screenshot}
            
            result = self._run_async(_get_url())
            self._update_action_result(action, result={"url": result["url"]})
            
            logger.info(f"Current URL: {result['url']}")
            
            return {
                "success": True,
                "url": result["url"],
                "screenshot": result["screenshot"]
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting current URL: {error_msg}")
            self._update_action_result(action, error=error_msg)
            
            # Try to get a screenshot anyway
            try:
                screenshot = self._run_async(self._get_screenshot())
            except:
                screenshot = ""
                
            return {
                "success": False,
                "error": error_msg,
                "screenshot": screenshot
            }
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """
        Get the action history
        
        Returns:
            List of actions as dictionaries
        """
        return [
            {
                "action_type": action.action_type,
                "params": action.params,
                "result": action.result,
                "error": action.error,
                "timestamp": action.timestamp
            }
            for action in self._action_history
        ]
    
    def clear_action_history(self) -> None:
        """Clear the action history"""
        self._action_history = []
        logger.debug("Cleared browser action history")
