#!/usr/bin/env python3
"""
Production Load Testing Script for TimesFM API
Simulates realistic production workloads with ETT and stock data patterns.
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
import pandas as pd
import argparse
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    api_url: str = "http://localhost:8000"
    concurrent_users: int = 10
    requests_per_user: int = 100
    ramp_up_time: int = 30  # seconds
    test_duration: int = 300  # seconds
    think_time: float = 0.5  # seconds between requests
    timeout: int = 120  # seconds per request
    output_dir: str = "./load_test_results"

@dataclass
class TestResult:
    """Individual test result."""
    request_id: str
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    request_size: int = 0
    response_size: int = 0
    model_size: str = "200M"
    input_length: int = 0
    horizon: int = 48
    cached: bool = False

@dataclass
class TestSummary:
    """Summary of test results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    cache_hit_rate: float
    total_data_sent: int
    total_data_received: int

class DataGenerator:
    """Generates realistic test data for load testing."""

    @staticmethod
    def generate_ett_data(length: int = 512) -> List[float]:
        """Generate ETT (Electricity Transformer Temperature) like data."""
        # Create realistic patterns: daily seasonality, weekly patterns, and trends
        t = np.arange(length)

        # Daily seasonality (24-hour pattern)
        daily = 5 * np.sin(2 * np.pi * t / 24)

        # Weekly pattern
        weekly = 3 * np.sin(2 * np.pi * t / (24 * 7))

        # Trend
        trend = 0.01 * t

        # Noise
        noise = np.random.normal(0, 2, length)

        # Combine and scale to realistic temperature range
        data = 20 + daily + weekly + trend + noise

        return data.tolist()

    @staticmethod
    def generate_stock_data(length: int = 512) -> List[float]:
        """Generate stock price like data with realistic patterns."""
        # Geometric Brownian Motion with mean reversion
        dt = 1/24  # hourly
        mu = 0.05  # drift
        sigma = 0.2  # volatility
        theta = 0.1  # mean reversion speed
        mean_price = 100

        prices = [mean_price]

        for i in range(1, length):
            # Mean reversion term
            mr_term = theta * (mean_price - prices[-1]) * dt

            # Random walk term
            random_term = mu * dt + sigma * np.sqrt(dt) * np.random.normal()

            new_price = prices[-1] * (1 + mr_term + random_term)
            prices.append(max(new_price, 1))  # Ensure positive prices

        return prices

    @staticmethod
    def generate_synthetic_data(length: int = 512, pattern: str = "mixed") -> List[float]:
        """Generate synthetic time series data with various patterns."""
        t = np.arange(length)

        if pattern == "trend":
            data = 100 + 0.5 * t + np.random.normal(0, 10, length)
        elif pattern == "seasonal":
            data = 100 + 20 * np.sin(2 * np.pi * t / 24) + 10 * np.sin(2 * np.pi * t / 168)
            data += np.random.normal(0, 5, length)
        elif pattern == "volatile":
            base = 100 + np.cumsum(np.random.normal(0, 5, length))
            volatility = np.abs(np.random.normal(0, 10, length))
            data = base + volatility
        else:  # mixed
            trend = 0.2 * t
            seasonal = 15 * np.sin(2 * np.pi * t / 24) + 8 * np.sin(2 * np.pi * t / 168)
            noise = np.random.normal(0, 12, length)
            data = 100 + trend + seasonal + noise

        return data.tolist()

class LoadTester:
    """Main load testing class."""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[TestResult] = []
        self.start_time: Optional[float] = None
        self.data_generator = DataGenerator()

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool size
            limit_per_host=50,
            keepalive_timeout=60
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate test data for a single request."""
        # Choose data pattern randomly
        patterns = ["ett", "stock", "synthetic_trend", "synthetic_seasonal", "synthetic_volatile"]
        pattern = np.random.choice(patterns)

        if pattern == "ett":
            data = self.data_generator.generate_ett_data()
        elif pattern == "stock":
            data = self.data_generator.generate_stock_data()
        elif pattern == "synthetic_trend":
            data = self.data_generator.generate_synthetic_data(pattern="trend")
        elif pattern == "synthetic_seasonal":
            data = self.data_generator.generate_synthetic_data(pattern="seasonal")
        else:  # synthetic_volatile
            data = self.data_generator.generate_synthetic_data(pattern="volatile")

        # Vary horizon lengths
        horizon = np.random.choice([24, 48, 96, 168, 336], p=[0.3, 0.4, 0.15, 0.1, 0.05])

        # Vary quantiles
        quantile_options = [
            [0.1, 0.5, 0.9],
            [0.05, 0.5, 0.95],
            [0.25, 0.5, 0.75],
            [0.1, 0.2, 0.5, 0.8, 0.9]
        ]
        quantiles = np.random.choice(quantile_options)

        return {
            "data": data,
            "horizon": horizon,
            "quantiles": quantiles,
            "num_samples": 1,
            "model_size": "200M"
        }

    async def _single_request(self, user_id: int, request_id: int) -> TestResult:
        """Execute a single API request."""
        request_start = time.time()

        # Generate test data
        test_data = self._generate_test_data()
        request_size = len(json.dumps(test_data).encode())

        try:
            async with self.session.post(
                f"{self.config.api_url}/forecast",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                response_text = await response.text()
                response_size = len(response_text.encode())
                response_time = time.time() - request_start

                if response.status == 200:
                    response_data = json.loads(response_text)
                    result = TestResult(
                        request_id=f"user_{user_id}_req_{request_id}",
                        status_code=response.status,
                        response_time=response_time,
                        success=True,
                        request_size=request_size,
                        response_size=response_size,
                        model_size=response_data.get("metadata", {}).get("model_size", "200M"),
                        input_length=len(test_data["data"]),
                        horizon=test_data["horizon"],
                        cached=response_data.get("metadata", {}).get("cached", False)
                    )
                else:
                    result = TestResult(
                        request_id=f"user_{user_id}_req_{request_id}",
                        status_code=response.status,
                        response_time=response_time,
                        success=False,
                        error_message=response_text[:200],
                        request_size=request_size,
                        response_size=response_size,
                        input_length=len(test_data["data"]),
                        horizon=test_data["horizon"]
                    )

                return result

        except Exception as e:
            response_time = time.time() - request_start
            return TestResult(
                request_id=f"user_{user_id}_req_{request_id}",
                status_code=0,
                response_time=response_time,
                success=False,
                error_message=str(e),
                request_size=request_size,
                input_length=len(test_data["data"]),
                horizon=test_data["horizon"]
            )

    async def _user_simulation(self, user_id: int, requests_to_make: int, ramp_up_delay: float = 0):
        """Simulate a single user making requests."""
        await asyncio.sleep(ramp_up_delay)

        for request_id in range(requests_to_make):
            # Check if we should stop based on test duration
            if self.start_time and (time.time() - self.start_time) > self.config.test_duration:
                break

            # Make request
            result = await self._single_request(user_id, request_id)
            self.results.append(result)

            # Think time between requests
            if self.config.think_time > 0:
                await asyncio.sleep(self.config.think_time)

    async def run_test(self) -> TestSummary:
        """Run the full load test."""
        logger.info(f"Starting load test with {self.config.concurrent_users} users")
        logger.info(f"Each user will make {self.config.requests_per_user} requests")
        logger.info(f"Test will run for max {self.config.test_duration} seconds")

        self.start_time = time.time()
        self.results = []

        # Calculate ramp-up delay per user
        ramp_up_delay = self.config.ramp_up_time / self.config.concurrent_users

        # Create user tasks
        tasks = []
        for user_id in range(self.config.concurrent_users):
            task = asyncio.create_task(
                self._user_simulation(
                    user_id=user_id,
                    requests_to_make=self.config.requests_per_user,
                    ramp_up_delay=user_id * ramp_up_delay
                )
            )
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - self.start_time

        # Calculate summary
        summary = self._calculate_summary(total_time)

        logger.info(f"Load test completed in {total_time:.2f} seconds")
        logger.info(f"Total requests: {summary.total_requests}")
        logger.info(f"Success rate: {(1 - summary.error_rate) * 100:.2f}%")
        logger.info(f"Average response time: {summary.avg_response_time:.3f}s")
        logger.info(f"95th percentile: {summary.p95_response_time:.3f}s")
        logger.info(f"Requests per second: {summary.requests_per_second:.2f}")

        return summary

    def _calculate_summary(self, total_time: float) -> TestSummary:
        """Calculate test summary statistics."""
        if not self.results:
            raise ValueError("No results to summarize")

        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]

        response_times = [r.response_time for r in successful_results]

        summary = TestSummary(
            total_requests=len(self.results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p50_response_time=np.percentile(response_times, 50) if response_times else 0,
            p95_response_time=np.percentile(response_times, 95) if response_times else 0,
            p99_response_time=np.percentile(response_times, 99) if response_times else 0,
            requests_per_second=len(self.results) / total_time if total_time > 0 else 0,
            error_rate=len(failed_results) / len(self.results) if self.results else 0,
            cache_hit_rate=sum(1 for r in successful_results if r.cached) / len(successful_results) if successful_results else 0,
            total_data_sent=sum(r.request_size for r in self.results),
            total_data_received=sum(r.response_size for r in self.results)
        )

        return summary

    def save_results(self, summary: TestSummary):
        """Save test results to files."""
        # Save summary
        summary_file = Path(self.config.output_dir) / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)

        # Save detailed results
        results_file = Path(self.config.output_dir) / "detailed_results.json"
        results_data = [asdict(r) for r in self.results]
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Generate visualizations
        self._generate_plots()

        logger.info(f"Results saved to {self.config.output_dir}")

    def _generate_plots(self):
        """Generate performance plots."""
        if not self.results:
            return

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([asdict(r) for r in self.results])
        successful_df = df[df['success'] == True]

        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('TimesFM API Load Test Results', fontsize=16)

        # 1. Response time over time
        if not successful_df.empty:
            successful_df['timestamp'] = range(len(successful_df))
            axes[0, 0].plot(successful_df['timestamp'], successful_df['response_time'],
                          alpha=0.6, markersize=2)
            axes[0, 0].set_title('Response Time Over Time')
            axes[0, 0].set_xlabel('Request Number')
            axes[0, 0].set_ylabel('Response Time (s)')
            axes[0, 0].grid(True)

        # 2. Response time distribution
        if not successful_df.empty:
            axes[0, 1].hist(successful_df['response_time'], bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Response Time Distribution')
            axes[0, 1].set_xlabel('Response Time (s)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)

        # 3. Status code distribution
        status_counts = df['status_code'].value_counts()
        axes[1, 0].bar(status_counts.index.astype(str), status_counts.values)
        axes[1, 0].set_title('Status Code Distribution')
        axes[1, 0].set_xlabel('Status Code')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True)

        # 4. Response time by input length
        if not successful_df.empty:
            axes[1, 1].scatter(successful_df['input_length'], successful_df['response_time'],
                             alpha=0.6, s=10)
            axes[1, 1].set_title('Response Time vs Input Length')
            axes[1, 1].set_xlabel('Input Length')
            axes[1, 1].set_ylabel('Response Time (s)')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plot_file = Path(self.config.output_dir) / "performance_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Generate performance report
        self._generate_report()

    def _generate_report(self):
        """Generate a detailed performance report."""
        summary = self._calculate_summary(time.time() - self.start_time)

        report_file = Path(self.config.output_dir) / "performance_report.md"

        with open(report_file, 'w') as f:
            f.write("# TimesFM API Load Test Report\n\n")
            f.write(f"**Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**API URL:** {self.config.api_url}\n\n")

            f.write("## Test Configuration\n\n")
            f.write(f"- Concurrent Users: {self.config.concurrent_users}\n")
            f.write(f"- Requests per User: {self.config.requests_per_user}\n")
            f.write(f"- Ramp-up Time: {self.config.ramp_up_time}s\n")
            f.write(f"- Test Duration: {self.config.test_duration}s\n")
            f.write(f"- Think Time: {self.config.think_time}s\n\n")

            f.write("## Test Results\n\n")
            f.write(f"- **Total Requests:** {summary.total_requests:,}\n")
            f.write(f"- **Successful Requests:** {summary.successful_requests:,}\n")
            f.write(f"- **Failed Requests:** {summary.failed_requests:,}\n")
            f.write(f"- **Success Rate:** {(1 - summary.error_rate) * 100:.2f}%\n")
            f.write(f"- **Requests per Second:** {summary.requests_per_second:.2f}\n")
            f.write(f"- **Cache Hit Rate:** {summary.cache_hit_rate * 100:.2f}%\n\n")

            f.write("## Response Time Statistics\n\n")
            f.write(f"- **Average:** {summary.avg_response_time:.3f}s\n")
            f.write(f"- **Minimum:** {summary.min_response_time:.3f}s\n")
            f.write(f"- **Maximum:** {summary.max_response_time:.3f}s\n")
            f.write(f"- **50th Percentile:** {summary.p50_response_time:.3f}s\n")
            f.write(f"- **95th Percentile:** {summary.p95_response_time:.3f}s\n")
            f.write(f"- **99th Percentile:** {summary.p99_response_time:.3f}s\n\n")

            f.write("## Data Transfer\n\n")
            f.write(f"- **Total Data Sent:** {summary.total_data_sent / 1024 / 1024:.2f} MB\n")
            f.write(f"- **Total Data Received:** {summary.total_data_received / 1024 / 1024:.2f} MB\n\n")

            # Error analysis
            if summary.failed_requests > 0:
                f.write("## Error Analysis\n\n")
                error_results = [r for r in self.results if not r.success]
                error_by_status = {}
                for result in error_results:
                    status = result.status_code
                    if status not in error_by_status:
                        error_by_status[status] = []
                    error_by_status[status].append(result.error_message)

                for status, messages in error_by_status.items():
                    f.write(f"### Status Code {status}\n")
                    f.write(f"Count: {len(messages)}\n")
                    if messages:
                        f.write(f"Sample Error: {messages[0][:100]}...\n")
                    f.write("\n")

        logger.info(f"Performance report generated: {report_file}")

async def main():
    """Main function to run load tests."""
    parser = argparse.ArgumentParser(description="TimesFM API Load Testing")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="API base URL")
    parser.add_argument("--users", type=int, default=10,
                       help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=100,
                       help="Requests per user")
    parser.add_argument("--duration", type=int, default=300,
                       help="Test duration in seconds")
    parser.add_argument("--ramp-up", type=int, default=30,
                       help="Ramp-up time in seconds")
    parser.add_argument("--output", default="./load_test_results",
                       help="Output directory")
    parser.add_argument("--think-time", type=float, default=0.5,
                       help="Think time between requests")

    args = parser.parse_args()

    # Create configuration
    config = LoadTestConfig(
        api_url=args.url,
        concurrent_users=args.users,
        requests_per_user=args.requests,
        test_duration=args.duration,
        ramp_up_time=args.ramp_up,
        output_dir=args.output,
        think_time=args.think_time
    )

    # Run load test
    async with LoadTester(config) as tester:
        summary = await tester.run_test()
        tester.save_results(summary)

    # Print summary
    print("\n" + "="*50)
    print("LOAD TEST SUMMARY")
    print("="*50)
    print(f"Total Requests: {summary.total_requests:,}")
    print(f"Success Rate: {(1 - summary.error_rate) * 100:.2f}%")
    print(f"Average Response Time: {summary.avg_response_time:.3f}s")
    print(f"95th Percentile: {summary.p95_response_time:.3f}s")
    print(f"Requests/sec: {summary.requests_per_second:.2f}")
    print(f"Cache Hit Rate: {summary.cache_hit_rate * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())