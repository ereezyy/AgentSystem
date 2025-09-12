"""
Email Module
-----------
Provides email capabilities for the agent
"""

import os
import time
import base64
import smtplib
import imaplib
import email
import email.utils
import email.header
import email.mime.text
import email.mime.multipart
import email.mime.application
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Local imports
from AgentSystem.utils.logger import get_logger
from AgentSystem.utils.env_loader import get_env

# Get module logger
logger = get_logger("modules.email")


@dataclass
class EmailAttachment:
    """Represents an email attachment"""
    filename: str
    content_type: str
    content: bytes
    
    @classmethod
    def from_file(cls, file_path: str, content_type: Optional[str] = None) -> 'EmailAttachment':
        """
        Create an attachment from a file
        
        Args:
            file_path: Path to the file
            content_type: Content type (if None, will be guessed)
            
        Returns:
            Email attachment
        """
        import mimetypes
        
        if not content_type:
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = 'application/octet-stream'
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        filename = Path(file_path).name
        
        return cls(
            filename=filename,
            content_type=content_type,
            content=content
        )


@dataclass
class Email:
    """Represents an email message"""
    subject: str
    body: str
    sender: Optional[str] = None
    recipients: List[str] = field(default_factory=list)
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    attachments: List[EmailAttachment] = field(default_factory=list)
    html_body: Optional[str] = None
    message_id: Optional[str] = None
    thread_id: Optional[str] = None
    date: Optional[datetime] = None
    
    def add_attachment(self, attachment: EmailAttachment) -> None:
        """
        Add an attachment to the email
        
        Args:
            attachment: Email attachment
        """
        self.attachments.append(attachment)
    
    def add_attachment_from_file(self, file_path: str, content_type: Optional[str] = None) -> None:
        """
        Add an attachment from a file
        
        Args:
            file_path: Path to the file
            content_type: Content type (if None, will be guessed)
        """
        attachment = EmailAttachment.from_file(file_path, content_type)
        self.add_attachment(attachment)
    
    def to_mime_message(self) -> email.mime.multipart.MIMEMultipart:
        """
        Convert to a MIME message
        
        Returns:
            MIME message
        """
        msg = email.mime.multipart.MIMEMultipart()
        msg['Subject'] = self.subject
        
        if self.sender:
            msg['From'] = self.sender
            
        if self.recipients:
            msg['To'] = ', '.join(self.recipients)
            
        if self.cc:
            msg['Cc'] = ', '.join(self.cc)
        
        # Add plain text body
        if self.body:
            text_part = email.mime.text.MIMEText(self.body, 'plain')
            msg.attach(text_part)
        
        # Add HTML body if provided
        if self.html_body:
            html_part = email.mime.text.MIMEText(self.html_body, 'html')
            msg.attach(html_part)
        
        # Add attachments
        for attachment in self.attachments:
            part = email.mime.application.MIMEApplication(attachment.content)
            part.add_header('Content-Disposition', 'attachment', filename=attachment.filename)
            part.add_header('Content-Type', attachment.content_type)
            msg.attach(part)
        
        return msg
    
    @classmethod
    def from_mime_message(cls, message: email.message.Message) -> 'Email':
        """
        Create an email from a MIME message
        
        Args:
            message: MIME message
            
        Returns:
            Email
        """
        # Extract subject
        subject, encoding = email.header.decode_header(message.get('Subject', ''))[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or 'utf-8', errors='replace')
        
        # Extract sender
        from_header = message.get('From', '')
        if from_header:
            sender = email.utils.parseaddr(from_header)[1]
        else:
            sender = None
        
        # Extract recipients
        to_header = message.get('To', '')
        if to_header:
            recipients = [addr[1] for addr in email.utils.getaddresses([to_header])]
        else:
            recipients = []
        
        # Extract CC
        cc_header = message.get('Cc', '')
        if cc_header:
            cc = [addr[1] for addr in email.utils.getaddresses([cc_header])]
        else:
            cc = []
        
        # Extract date
        date_header = message.get('Date')
        if date_header:
            date_tuple = email.utils.parsedate_tz(date_header)
            if date_tuple:
                date = datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
            else:
                date = None
        else:
            date = None
        
        # Extract message ID and thread ID
        message_id = message.get('Message-ID')
        thread_id = message.get('Thread-Topic')  # Not all email providers use this
        
        # Extract body parts and attachments
        body = ""
        html_body = None
        attachments = []
        
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                
                # Extract attachments
                if 'attachment' in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        # Decode filename if needed
                        if isinstance(filename, str):
                            filename_parts = email.header.decode_header(filename)
                            if filename_parts and isinstance(filename_parts[0][0], bytes):
                                filename = filename_parts[0][0].decode(filename_parts[0][1] or 'utf-8', errors='replace')
                        
                        content = part.get_payload(decode=True)
                        
                        attachment = EmailAttachment(
                            filename=filename,
                            content_type=content_type,
                            content=content
                        )
                        
                        attachments.append(attachment)
                        
                # Extract text body
                elif content_type == 'text/plain' and 'attachment' not in content_disposition:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        body = payload.decode(charset, errors='replace')
                    except:
                        body = payload.decode('utf-8', errors='replace')
                        
                # Extract HTML body
                elif content_type == 'text/html' and 'attachment' not in content_disposition:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        html_body = payload.decode(charset, errors='replace')
                    except:
                        html_body = payload.decode('utf-8', errors='replace')
        else:
            # Handle non-multipart messages
            content_type = message.get_content_type()
            if content_type == 'text/plain':
                payload = message.get_payload(decode=True)
                charset = message.get_content_charset() or 'utf-8'
                try:
                    body = payload.decode(charset, errors='replace')
                except:
                    body = payload.decode('utf-8', errors='replace')
            elif content_type == 'text/html':
                payload = message.get_payload(decode=True)
                charset = message.get_content_charset() or 'utf-8'
                try:
                    html_body = payload.decode(charset, errors='replace')
                except:
                    html_body = payload.decode('utf-8', errors='replace')
        
        return cls(
            subject=subject or "",
            body=body,
            sender=sender,
            recipients=recipients,
            cc=cc,
            attachments=attachments,
            html_body=html_body,
            message_id=message_id,
            thread_id=thread_id,
            date=date
        )


class EmailModule:
    """
    Email module for sending and receiving emails
    
    Supports SMTP for sending and IMAP for receiving
    """
    
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        imap_server: Optional[str] = None,
        imap_port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = True,
        default_sender: Optional[str] = None,
        memory_manager = None
    ):
        """
        Initialize the email module
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            imap_server: IMAP server address
            imap_port: IMAP server port
            username: Email account username
            password: Email account password
            use_ssl: Whether to use SSL/TLS
            default_sender: Default sender email address
            memory_manager: Memory manager for storing emails
        """
        # SMTP configuration
        self.smtp_server = smtp_server or get_env("SMTP_SERVER")
        self.smtp_port = smtp_port or int(get_env("SMTP_PORT") or 0) or (465 if use_ssl else 587)
        
        # IMAP configuration
        self.imap_server = imap_server or get_env("IMAP_SERVER")
        self.imap_port = imap_port or int(get_env("IMAP_PORT") or 0) or (993 if use_ssl else 143)
        
        # Authentication
        self.username = username or get_env("EMAIL_USERNAME")
        self.password = password or get_env("EMAIL_PASSWORD")
        
        # Other settings
        self.use_ssl = use_ssl
        self.default_sender = default_sender or self.username
        self.memory_manager = memory_manager
        
        # Connection objects
        self._smtp = None
        self._imap = None
        
        logger.debug("Email module initialized")
    
    def _connect_smtp(self) -> None:
        """Connect to the SMTP server"""
        if self._smtp:
            return
            
        try:
            if self.use_ssl:
                self._smtp = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                self._smtp = smtplib.SMTP(self.smtp_server, self.smtp_port)
                self._smtp.starttls()
            
            self._smtp.login(self.username, self.password)
            logger.debug(f"Connected to SMTP server {self.smtp_server}:{self.smtp_port}")
            
        except Exception as e:
            logger.error(f"Error connecting to SMTP server: {e}")
            self._smtp = None
            raise
    
    def _disconnect_smtp(self) -> None:
        """Disconnect from the SMTP server"""
        if self._smtp:
            try:
                self._smtp.quit()
            except:
                pass
            self._smtp = None
            logger.debug("Disconnected from SMTP server")
    
    def _connect_imap(self) -> None:
        """Connect to the IMAP server"""
        if self._imap:
            return
            
        try:
            if self.use_ssl:
                self._imap = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            else:
                self._imap = imaplib.IMAP4(self.imap_server, self.imap_port)
                self._imap.starttls()
            
            self._imap.login(self.username, self.password)
            logger.debug(f"Connected to IMAP server {self.imap_server}:{self.imap_port}")
            
        except Exception as e:
            logger.error(f"Error connecting to IMAP server: {e}")
            self._imap = None
            raise
    
    def _disconnect_imap(self) -> None:
        """Disconnect from the IMAP server"""
        if self._imap:
            try:
                self._imap.logout()
            except:
                pass
            self._imap = None
            logger.debug("Disconnected from IMAP server")
    
    def send_email(self, email_obj: Email) -> bool:
        """
        Send an email
        
        Args:
            email_obj: Email to send
            
        Returns:
            Success flag
        """
        # Set sender if not already set
        if not email_obj.sender:
            email_obj.sender = self.default_sender
        
        # Check recipients
        all_recipients = email_obj.recipients + email_obj.cc + email_obj.bcc
        if not all_recipients:
            logger.error("Cannot send email without recipients")
            return False
        
        try:
            # Connect to the SMTP server
            self._connect_smtp()
            
            # Convert to MIME message
            mime_msg = email_obj.to_mime_message()
            
            # Send the email
            self._smtp.send_message(
                mime_msg,
                from_addr=email_obj.sender,
                to_addrs=all_recipients
            )
            
            logger.info(f"Sent email '{email_obj.subject}' to {', '.join(all_recipients)}")
            
            # Store in memory if we have a memory manager
            if self.memory_manager:
                self.memory_manager.add(
                    content={
                        "action": "send",
                        "subject": email_obj.subject,
                        "body": email_obj.body,
                        "recipients": all_recipients,
                        "timestamp": time.time()
                    },
                    memory_type="email",
                    importance=0.7,
                    metadata={
                        "module": "email",
                        "action": "send",
                        "subject": email_obj.subject
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
            
        finally:
            # Disconnect from the SMTP server
            self._disconnect_smtp()
    
    def get_emails(
        self,
        folder: str = "INBOX",
        limit: int = 10,
        unread_only: bool = False,
        search_criteria: Optional[str] = None
    ) -> List[Email]:
        """
        Get emails from a folder
        
        Args:
            folder: Email folder
            limit: Maximum number of emails to fetch
            unread_only: Whether to fetch only unread emails
            search_criteria: Additional IMAP search criteria
            
        Returns:
            List of emails
        """
        try:
            # Connect to the IMAP server
            self._connect_imap()
            
            # Select the folder
            status, data = self._imap.select(folder)
            if status != 'OK':
                logger.error(f"Error selecting folder {folder}: {data}")
                return []
            
            # Build search criteria
            search_query = "ALL"
            if unread_only:
                search_query = "UNSEEN"
                
            if search_criteria:
                search_query = f"({search_query} {search_criteria})"
            
            # Search for emails
            status, data = self._imap.search(None, search_query)
            if status != 'OK':
                logger.error(f"Error searching for emails: {data}")
                return []
            
            # Get message IDs
            message_ids = data[0].split()
            if not message_ids:
                logger.info(f"No emails found in folder {folder}")
                return []
            
            # Limit the number of emails
            message_ids = message_ids[-limit:] if limit > 0 else message_ids
            
            # Fetch emails
            emails = []
            for msg_id in message_ids:
                status, data = self._imap.fetch(msg_id, '(RFC822)')
                if status != 'OK':
                    logger.warning(f"Error fetching email {msg_id}: {data}")
                    continue
                
                # Parse the email
                message = email.message_from_bytes(data[0][1])
                email_obj = Email.from_mime_message(message)
                
                # Store in memory if we have a memory manager
                if self.memory_manager:
                    self.memory_manager.add(
                        content={
                            "action": "receive",
                            "subject": email_obj.subject,
                            "body": email_obj.body,
                            "sender": email_obj.sender,
                            "timestamp": time.time()
                        },
                        memory_type="email",
                        importance=0.7,
                        metadata={
                            "module": "email",
                            "action": "receive",
                            "subject": email_obj.subject,
                            "sender": email_obj.sender
                        }
                    )
                
                emails.append(email_obj)
            
            logger.info(f"Retrieved {len(emails)} emails from folder {folder}")
            return emails
            
        except Exception as e:
            logger.error(f"Error retrieving emails: {e}")
            return []
            
        finally:
            # Disconnect from the IMAP server
            self._disconnect_imap()
    
    def mark_as_read(self, folder: str = "INBOX", email_subjects: List[str] = None) -> bool:
        """
        Mark emails as read
        
        Args:
            folder: Email folder
            email_subjects: List of email subjects to mark as read (if None, mark all)
            
        Returns:
            Success flag
        """
        try:
            # Connect to the IMAP server
            self._connect_imap()
            
            # Select the folder
            status, data = self._imap.select(folder)
            if status != 'OK':
                logger.error(f"Error selecting folder {folder}: {data}")
                return False
            
            # Find unread emails
            status, data = self._imap.search(None, "UNSEEN")
            if status != 'OK':
                logger.error("Error searching for unread emails")
                return False
            
            message_ids = data[0].split()
            if not message_ids:
                logger.info("No unread emails found")
                return True
            
            # If specific subjects are provided, filter messages
            if email_subjects:
                filtered_ids = []
                for msg_id in message_ids:
                    status, data = self._imap.fetch(msg_id, '(BODY.PEEK[HEADER.FIELDS (SUBJECT)])')
                    if status != 'OK':
                        continue
                    
                    subject_data = data[0][1].decode('utf-8', errors='replace')
                    for subject in email_subjects:
                        if subject in subject_data:
                            filtered_ids.append(msg_id)
                            break
                
                message_ids = filtered_ids
            
            # Mark emails as read
            for msg_id in message_ids:
                self._imap.store(msg_id, '+FLAGS', '\\Seen')
            
            logger.info(f"Marked {len(message_ids)} emails as read")
            return True
            
        except Exception as e:
            logger.error(f"Error marking emails as read: {e}")
            return False
            
        finally:
            # Disconnect from the IMAP server
            self._disconnect_imap()
    
    def create_email(
        self,
        subject: str,
        body: str,
        recipients: List[str],
        cc: List[str] = None,
        bcc: List[str] = None,
        html_body: str = None,
        attachments: List[str] = None
    ) -> Email:
        """
        Create an email
        
        Args:
            subject: Email subject
            body: Email body
            recipients: List of recipients
            cc: List of CC recipients
            bcc: List of BCC recipients
            html_body: HTML version of the email body
            attachments: List of attachment file paths
            
        Returns:
            Email object
        """
        email_obj = Email(
            subject=subject,
            body=body,
            sender=self.default_sender,
            recipients=recipients,
            cc=cc or [],
            bcc=bcc or [],
            html_body=html_body
        )
        
        # Add attachments
        if attachments:
            for file_path in attachments:
                email_obj.add_attachment_from_file(file_path)
        
        return email_obj
