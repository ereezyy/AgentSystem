import re

filepath = 'AgentSystem/core/memory.py'

with open(filepath, 'r') as f:
    content = f.read()

# Pattern matching the duplicate block
duplicate_block = r'    def __enter__\(self\):\n        """Context manager entry"""\n        return self\n\n    def __exit__\(self, exc_type, exc_val, exc_tb\):\n        """Context manager exit"""\n        self\.close\(\)'

# Replace the second occurrence with empty string (or rather, replace all with one instance)
# Since re.sub replaces all occurrences, we can just replace it with a single instance.
# Wait, if I replace all with one, it might still leave one if they are adjacent.

# Let's try to find the specific location.
parts = content.split('    def __del__(self):')
header = parts[0]
footer = parts[1]

# In the header, remove the second occurrence of the block if it exists
# Actually, the duplicate is right before __del__.

# Let's just rewrite the section.
pattern_section = r'    def close\(self\):\n        """Close the database connection"""\n        if hasattr\(self, "_conn"\) and self._conn:\n            self._conn\.close\(\)\n            self._conn = None\n\n    def __enter__\(self\):\n        """Context manager entry"""\n        return self\n\n    def __exit__\(self, exc_type, exc_val, exc_tb\):\n        """Context manager exit"""\n        self\.close\(\)\n\n    def __enter__\(self\):\n        """Context manager entry"""\n        return self\n\n    def __exit__\(self, exc_type, exc_val, exc_tb\):\n        """Context manager exit"""\n        self\.close\(\)'

replacement_section = r'    def close(self):\n        """Close the database connection"""\n        if hasattr(self, "_conn") and self._conn:\n            self._conn.close()\n            self._conn = None\n\n    def __enter__(self):\n        """Context manager entry"""\n        return self\n\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        """Context manager exit"""\n        self.close()'

new_content = re.sub(pattern_section, replacement_section, content, flags=re.DOTALL)

# Also check for logs in the diff? No, I already deleted logs. The review might be stale or referring to diff.
# Wait, did I commit the deletion of logs?
# The review said "The patch includes logs/agent_20260222.log".
# I deleted them in bash, but did I submit that?
# The  tool takes . If I reuse the branch name, it updates the PR.
# I haven't submitted the deletion yet. I need to include it in the next submit.

with open(filepath, 'w') as f:
    f.write(new_content)
