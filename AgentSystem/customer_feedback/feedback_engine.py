"""
Customer Feedback and Feature Request Engine

This module handles the collection, processing, and management of customer feedback
and feature requests. It includes mechanisms for feedback submission, categorization,
prioritization, and integration with the product development lifecycle.
"""

import uuid
import json
import asyncio
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class FeedbackEngine:
    def __init__(self, db_connection=None, notification_service=None):
        """Initialize the Feedback Engine with database and notification services."""
        self.db_connection = db_connection
        self.notification_service = notification_service
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.feedback_store: Dict[str, dict] = {}
        self.feature_requests: Dict[str, dict] = {}
        self.clusters = None

    async def submit_feedback(self, tenant_id: str, user_id: str, feedback_data: dict) -> str:
        """
        Submit customer feedback for processing.

        Args:
            tenant_id: Identifier for the tenant
            user_id: Identifier for the user submitting feedback
            feedback_data: Dictionary containing feedback details

        Returns:
            feedback_id: Unique identifier for the submitted feedback
        """
        try:
            feedback_id = f"feedback_{uuid.uuid4().hex[:12]}"

            feedback_record = {
                'feedback_id': feedback_id,
                'tenant_id': tenant_id,
                'user_id': user_id,
                'title': feedback_data.get('title', ''),
                'description': feedback_data.get('description', ''),
                'category': feedback_data.get('category', 'general'),
                'rating': feedback_data.get('rating'),
                'sentiment': self._analyze_sentiment(feedback_data.get('description', '')),
                'status': 'submitted',
                'submitted_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'priority': 'medium',
                'related_feature_request': feedback_data.get('related_feature_request')
            }

            self.feedback_store[feedback_id] = feedback_record

            # Categorize and prioritize feedback
            self._categorize_feedback(feedback_record)
            await self._notify_relevant_teams(tenant_id, feedback_record)

            self.logger.info(f"Feedback submitted: {feedback_id} for tenant {tenant_id}")
            return feedback_id

        except Exception as e:
            self.logger.error(f"Error submitting feedback: {str(e)}")
            raise

    async def submit_feature_request(self, tenant_id: str, user_id: str, request_data: dict) -> str:
        """
        Submit a feature request for consideration.

        Args:
            tenant_id: Identifier for the tenant
            user_id: Identifier for the user submitting the request
            request_data: Dictionary containing feature request details

        Returns:
            request_id: Unique identifier for the submitted feature request
        """
        try:
            request_id = f"feature_{uuid.uuid4().hex[:12]}"

            feature_request = {
                'request_id': request_id,
                'tenant_id': tenant_id,
                'user_id': user_id,
                'title': request_data.get('title', ''),
                'description': request_data.get('description', ''),
                'use_case': request_data.get('use_case', ''),
                'priority': request_data.get('priority', 'medium'),
                'status': 'new',
                'votes': 1,
                'voters': {user_id: datetime.utcnow().isoformat()},
                'submitted_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'impact_score': self._calculate_impact_score(request_data),
                'implementation_complexity': self._estimate_implementation_complexity(request_data)
            }

            self.feature_requests[request_id] = feature_request

            # Notify product team
            await self._notify_product_team(tenant_id, feature_request)

            self.logger.info(f"Feature request submitted: {request_id} for tenant {tenant_id}")
            return request_id

        except Exception as e:
            self.logger.error(f"Error submitting feature request: {str(e)}")
            raise

    async def vote_feature_request(self, tenant_id: str, user_id: str, request_id: str) -> bool:
        """
        Vote on a feature request to increase its priority.

        Args:
            tenant_id: Identifier for the tenant
            user_id: Identifier for the user voting
            request_id: Identifier for the feature request

        Returns:
            bool: True if vote was successful, False otherwise
        """
        try:
            if request_id not in self.feature_requests:
                return False

            feature_request = self.feature_requests[request_id]

            if user_id not in feature_request['voters']:
                feature_request['votes'] += 1
                feature_request['voters'][user_id] = datetime.utcnow().isoformat()
                feature_request['updated_at'] = datetime.utcnow().isoformat()

                # Recalculate priority based on votes
                self._update_feature_priority(feature_request)

                self.logger.info(f"Vote added for feature request {request_id} by user {user_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error voting on feature request: {str(e)}")
            return False

    def get_feedback(self, feedback_id: str, tenant_id: str) -> Optional[dict]:
        """
        Retrieve feedback details.

        Args:
            feedback_id: Identifier for the feedback
            tenant_id: Identifier for the tenant

        Returns:
            dict: Feedback details if found and authorized, None otherwise
        """
        feedback = self.feedback_store.get(feedback_id)
        if feedback and feedback['tenant_id'] == tenant_id:
            return feedback
        return None

    def get_feature_request(self, request_id: str, tenant_id: str) -> Optional[dict]:
        """
        Retrieve feature request details.

        Args:
            request_id: Identifier for the feature request
            tenant_id: Identifier for the tenant

        Returns:
            dict: Feature request details if found and authorized, None otherwise
        """
        feature_request = self.feature_requests.get(request_id)
        if feature_request and feature_request['tenant_id'] == tenant_id:
            return feature_request
        return None

    def list_feedback(self, tenant_id: str, filters: dict = None) -> List[dict]:
        """
        List feedback submissions with optional filters.

        Args:
            tenant_id: Identifier for the tenant
            filters: Dictionary of filter criteria

        Returns:
            List[dict]: List of feedback records matching criteria
        """
        feedback_list = [f for f in self.feedback_store.values() if f['tenant_id'] == tenant_id]

        if filters:
            if 'category' in filters:
                feedback_list = [f for f in feedback_list if f['category'] == filters['category']]
            if 'status' in filters:
                feedback_list = [f for f in feedback_list if f['status'] == filters['status']]
            if 'priority' in filters:
                feedback_list = [f for f in feedback_list if f['priority'] == filters['priority']]
            if 'sentiment' in filters:
                feedback_list = [f for f in feedback_list if f['sentiment'] == filters['sentiment']]

        return sorted(feedback_list, key=lambda x: x['submitted_at'], reverse=True)

    def list_feature_requests(self, tenant_id: str, filters: dict = None) -> List[dict]:
        """
        List feature requests with optional filters.

        Args:
            tenant_id: Identifier for the tenant
            filters: Dictionary of filter criteria

        Returns:
            List[dict]: List of feature requests matching criteria
        """
        requests_list = [r for r in self.feature_requests.values() if r['tenant_id'] == tenant_id]

        if filters:
            if 'status' in filters:
                requests_list = [r for r in requests_list if r['status'] == filters['status']]
            if 'priority' in filters:
                requests_list = [r for r in requests_list if r['priority'] == filters['priority']]

        return sorted(requests_list, key=lambda x: (x['votes'], x['impact_score']), reverse=True)

    async def update_feedback_status(self, feedback_id: str, tenant_id: str, status: str, notes: str = '') -> bool:
        """
        Update the status of a feedback item.

        Args:
            feedback_id: Identifier for the feedback
            tenant_id: Identifier for the tenant
            status: New status for the feedback
            notes: Additional notes about the status change

        Returns:
            bool: True if update was successful, False otherwise
        """
        feedback = self.get_feedback(feedback_id, tenant_id)
        if not feedback:
            return False

        feedback['status'] = status
        feedback['updated_at'] = datetime.utcnow().isoformat()
        if notes:
            feedback['notes'] = feedback.get('notes', []) + [{'date': datetime.utcnow().isoformat(), 'text': notes}]

        await self._notify_status_change(tenant_id, feedback, status)
        return True

    async def update_feature_status(self, request_id: str, tenant_id: str, status: str, notes: str = '') -> bool:
        """
        Update the status of a feature request.

        Args:
            request_id: Identifier for the feature request
            tenant_id: Identifier for the tenant
            status: New status for the feature request
            notes: Additional notes about the status change

        Returns:
            bool: True if update was successful, False otherwise
        """
        feature_request = self.get_feature_request(request_id, tenant_id)
        if not feature_request:
            return False

        feature_request['status'] = status
        feature_request['updated_at'] = datetime.utcnow().isoformat()
        if notes:
            feature_request['notes'] = feature_request.get('notes', []) + [{'date': datetime.utcnow().isoformat(), 'text': notes}]

        await self._notify_feature_status_change(tenant_id, feature_request, status)
        return True

    def analyze_feedback_trends(self, tenant_id: str, time_range_days: int = 30) -> dict:
        """
        Analyze feedback trends and patterns.

        Args:
            tenant_id: Identifier for the tenant
            time_range_days: Number of days to analyze

        Returns:
            dict: Analysis results with trends and insights
        """
        cutoff_date = datetime.utcnow() - timedelta(days=time_range_days)
        feedback_list = [
            f for f in self.feedback_store.values()
            if f['tenant_id'] == tenant_id and datetime.fromisoformat(f['submitted_at']) >= cutoff_date
        ]

        if not feedback_list:
            return {"message": "No feedback data for the specified time range"}

        # Calculate metrics
        total_feedback = len(feedback_list)
        categories = {}
        sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
        priorities = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

        for feedback in feedback_list:
            categories[feedback['category']] = categories.get(feedback['category'], 0) + 1
            sentiments[feedback['sentiment']] += 1
            priorities[feedback['priority']] += 1

        # Identify trending issues
        trending_issues = self._identify_trending_issues(feedback_list)

        return {
            'total_feedback': total_feedback,
            'time_range_days': time_range_days,
            'category_distribution': categories,
            'sentiment_distribution': sentiments,
            'priority_distribution': priorities,
            'trending_issues': trending_issues,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }

    def analyze_feature_demand(self, tenant_id: str) -> dict:
        """
        Analyze demand and prioritization for features.

        Args:
            tenant_id: Identifier for the tenant

        Returns:
            dict: Analysis of feature request demand
        """
        requests_list = [r for r in self.feature_requests.values() if r['tenant_id'] == tenant_id]

        if not requests_list:
            return {"message": "No feature request data available"}

        # Calculate metrics
        total_requests = len(requests_list)
        status_dist = {'new': 0, 'under_review': 0, 'planned': 0, 'in_progress': 0, 'completed': 0, 'rejected': 0}
        priority_dist = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        total_votes = 0

        for request in requests_list:
            status_dist[request['status']] += 1
            priority_dist[request['priority']] += 1
            total_votes += request['votes']

        # Top requested features
        top_features = sorted(requests_list, key=lambda x: x['votes'], reverse=True)[:5]
        top_feature_summary = [
            {
                'request_id': f['request_id'],
                'title': f['title'],
                'votes': f['votes'],
                'impact_score': f['impact_score'],
                'status': f['status']
            }
            for f in top_features
        ]

        return {
            'total_requests': total_requests,
            'average_votes_per_request': total_votes / total_requests if total_requests > 0 else 0,
            'status_distribution': status_dist,
            'priority_distribution': priority_dist,
            'top_features': top_feature_summary,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }

    def _analyze_sentiment(self, text: str) -> str:
        """
        Analyze the sentiment of feedback text.

        Args:
            text: Text to analyze

        Returns:
            str: Sentiment classification (positive, neutral, negative)
        """
        if not text:
            return 'neutral'

        # Simple sentiment analysis based on keywords
        positive_words = {'good', 'great', 'excellent', 'happy', 'pleased', 'satisfied', 'awesome', 'fantastic'}
        negative_words = {'bad', 'poor', 'terrible', 'unhappy', 'dissatisfied', 'frustrated', 'awful', 'horrible'}

        tokens = word_tokenize(text.lower())
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)

        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        return 'neutral'

    def _categorize_feedback(self, feedback: dict) -> None:
        """
        Categorize feedback based on content analysis.

        Args:
            feedback: Feedback record to categorize
        """
        description = feedback.get('description', '').lower()
        title = feedback.get('title', '').lower()
        text = title + ' ' + description

        # Simple categorization based on keywords
        if any(word in text for word in ['bug', 'error', 'crash', 'issue', 'problem']):
            feedback['category'] = 'bug'
        elif any(word in text for word in ['feature', 'enhancement', 'suggest', 'idea', 'add']):
            feedback['category'] = 'feature_request'
        elif any(word in text for word in ['ui', 'interface', 'design', 'layout', 'ux']):
            feedback['category'] = 'user_interface'
        elif any(word in text for word in ['performance', 'slow', 'fast', 'speed', 'lag']):
            feedback['category'] = 'performance'
        elif any(word in text for word in ['support', 'help', 'assistance', 'ticket']):
            feedback['category'] = 'customer_support'

        # Update priority based on sentiment and category
        if feedback['sentiment'] == 'negative' and feedback['category'] == 'bug':
            feedback['priority'] = 'high'
        elif feedback['sentiment'] == 'negative':
            feedback['priority'] = 'medium'
        elif feedback['category'] == 'feature_request':
            feedback['priority'] = 'low'

    def _calculate_impact_score(self, request_data: dict) -> float:
        """
        Calculate impact score for a feature request.

        Args:
            request_data: Feature request data

        Returns:
            float: Impact score
        """
        # Simple impact scoring based on keywords and description length
        text = (request_data.get('title', '') + ' ' + request_data.get('description', '')).lower()
        impact_keywords = {'important', 'critical', 'essential', 'urgent', 'necessary', 'vital'}
        impact_count = sum(1 for word in text.split() if word in impact_keywords)

        description_length = len(request_data.get('description', '').split())

        # Normalize scores to 0-100 range
        keyword_score = min(impact_count * 10, 50)
        detail_score = min(description_length, 50)

        return keyword_score + detail_score

    def _estimate_implementation_complexity(self, request_data: dict) -> str:
        """
        Estimate implementation complexity for a feature.

        Args:
            request_data: Feature request data

        Returns:
            str: Complexity level (low, medium, high)
        """
        text = (request_data.get('title', '') + ' ' + request_data.get('description', '')).lower()

        high_complexity_keywords = {'system', 'architecture', 'infrastructure', 'integration', 'api', 'database', 'core'}
        medium_complexity_keywords = {'feature', 'functionality', 'module', 'component'}

        high_count = sum(1 for word in text.split() if word in high_complexity_keywords)
        medium_count = sum(1 for word in text.split() if word in medium_complexity_keywords)

        if high_count > 2:
            return 'high'
        elif medium_count > 1 or high_count > 0:
            return 'medium'
        return 'low'

    def _update_feature_priority(self, feature_request: dict) -> None:
        """
        Update feature request priority based on votes and impact.

        Args:
            feature_request: Feature request record
        """
        votes = feature_request['votes']
        impact_score = feature_request['impact_score']
        complexity = feature_request['implementation_complexity']

        # Calculate priority score
        priority_score = votes * 10 + impact_score

        if complexity == 'high':
            priority_score *= 0.7  # Reduce priority for highly complex features
        elif complexity == 'low':
            priority_score *= 1.2  # Increase priority for simple features

        if priority_score > 80:
            feature_request['priority'] = 'critical'
        elif priority_score > 50:
            feature_request['priority'] = 'high'
        elif priority_score > 20:
            feature_request['priority'] = 'medium'
        else:
            feature_request['priority'] = 'low'

    def _identify_trending_issues(self, feedback_list: List[dict]) -> List[dict]:
        """
        Identify trending issues from feedback using clustering.

        Args:
            feedback_list: List of feedback records

        Returns:
            List[dict]: List of trending issue summaries
        """
        if len(feedback_list) < 3:
            return []

        # Prepare text data for clustering
        texts = [f['title'] + ' ' + f['description'] for f in feedback_list]
        try:
            # Vectorize text data
            X = self.vectorizer.fit_transform(texts)

            # Perform K-means clustering
            num_clusters = min(5, len(feedback_list) // 2 + 1)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)

            # Group feedback by cluster
            cluster_groups = {}
            for idx, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(feedback_list[idx])

            # Identify trending issues based on cluster size and sentiment
            trending_issues = []
            for cluster_id, group in cluster_groups.items():
                if len(group) >= 2:  # Only consider clusters with multiple items
                    negative_count = sum(1 for f in group if f['sentiment'] == 'negative')
                    severity_score = len(group) * (1 + negative_count / max(1, len(group)))

                    # Extract representative terms
                    representative_text = ' '.join([f['title'] for f in group[:3]])
                    terms = [w for w in word_tokenize(representative_text.lower())
                            if w not in self.stop_words and len(w) > 3]
                    terms = [self.lemmatizer.lemmatize(t) for t in terms]
                    term_freq = {}
                    for term in terms:
                        term_freq[term] = term_freq.get(term, 0) + 1
                    key_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:3]
                    key_term_names = [t[0] for t in key_terms]

                    trending_issues.append({
                        'issue_id': f"trend_{cluster_id}",
                        'description': f"Related to: {', '.join(key_term_names)}",
                        'count': len(group),
                        'negative_percentage': (negative_count / len(group)) * 100,
                        'severity_score': severity_score,
                        'category': group[0]['category'],
                        'sample_feedback_ids': [f['feedback_id'] for f in group[:3]]
                    })

            return sorted(trending_issues, key=lambda x: x['severity_score'], reverse=True)[:3]

        except Exception as e:
            self.logger.error(f"Error identifying trending issues: {str(e)}")
            return []

    async def _notify_relevant_teams(self, tenant_id: str, feedback: dict) -> None:
        """
        Notify relevant teams about new feedback.

        Args:
            tenant_id: Identifier for the tenant
            feedback: Feedback record
        """
        if not self.notification_service:
            return

        try:
            # Determine recipient team based on category
            category = feedback['category']
            if category == 'bug':
                recipient = f"{tenant_id}:engineering-team"
                priority = 'high'
            elif category == 'feature_request':
                recipient = f"{tenant_id}:product-team"
                priority = 'medium'
            elif category == 'performance':
                recipient = f"{tenant_id}:devops-team"
                priority = 'high'
            elif category == 'user_interface':
                recipient = f"{tenant_id}:design-team"
                priority = 'medium'
            else:
                recipient = f"{tenant_id}:support-team"
                priority = 'low'

            message = {
                'title': f"New Feedback: {feedback['title']}",
                'body': f"Category: {category}\nSentiment: {feedback['sentiment']}\nPriority: {feedback['priority']}\nDescription: {feedback['description'][:200]}...",
                'metadata': {
                    'feedback_id': feedback['feedback_id'],
                    'tenant_id': tenant_id,
                    'category': category,
                    'priority': feedback['priority']
                }
            }

            await self.notification_service.send_notification(recipient, message, priority)

        except Exception as e:
            self.logger.error(f"Error notifying teams about feedback: {str(e)}")

    async def _notify_product_team(self, tenant_id: str, feature_request: dict) -> None:
        """
        Notify product team about new feature request.

        Args:
            tenant_id: Identifier for the tenant
            feature_request: Feature request record
        """
        if not self.notification_service:
            return

        try:
            recipient = f"{tenant_id}:product-team"
            message = {
                'title': f"New Feature Request: {feature_request['title']}",
                'body': f"Priority: {feature_request['priority']}\nImpact Score: {feature_request['impact_score']:.1f}\nComplexity: {feature_request['implementation_complexity']}\nDescription: {feature_request['description'][:200]}...",
                'metadata': {
                    'request_id': feature_request['request_id'],
                    'tenant_id': tenant_id,
                    'priority': feature_request['priority']
                }
            }

            await self.notification_service.send_notification(recipient, message, 'medium')

        except Exception as e:
            self.logger.error(f"Error notifying product team about feature request: {str(e)}")

    async def _notify_status_change(self, tenant_id: str, feedback: dict, new_status: str) -> None:
        """
        Notify relevant parties about feedback status changes.

        Args:
            tenant_id: Identifier for the tenant
            feedback: Feedback record
            new_status: New status value
        """
        if not self.notification_service:
            return

        try:
            # Notify submitting user
            user_id = feedback['user_id']
            recipient = f"{tenant_id}:user:{user_id}"
            message = {
                'title': f"Feedback Update: {feedback['title']}",
                'body': f"Your feedback has been updated to status: {new_status}",
                'metadata': {
                    'feedback_id': feedback['feedback_id'],
                    'tenant_id': tenant_id,
                    'new_status': new_status
                }
            }

            await self.notification_service.send_notification(recipient, message, 'low')

        except Exception as e:
            self.logger.error(f"Error notifying about feedback status change: {str(e)}")

    async def _notify_feature_status_change(self, tenant_id: str, feature_request: dict, new_status: str) -> None:
        """
        Notify relevant parties about feature request status changes.

        Args:
            tenant_id: Identifier for the tenant
            feature_request: Feature request record
            new_status: New status value
        """
        if not self.notification_service:
            return

        try:
            # Notify submitting user and voters
            user_ids = list(feature_request['voters'].keys())
            for user_id in user_ids[:10]:  # Limit to first 10 to avoid notification spam
                recipient = f"{tenant_id}:user:{user_id}"
                message = {
                    'title': f"Feature Request Update: {feature_request['title']}",
                    'body': f"The feature request you voted for has been updated to status: {new_status}",
                    'metadata': {
                        'request_id': feature_request['request_id'],
                        'tenant_id': tenant_id,
                        'new_status': new_status
                    }
                }

                await self.notification_service.send_notification(recipient, message, 'low')

        except Exception as e:
            self.logger.error(f"Error notifying about feature request status change: {str(e)}")
