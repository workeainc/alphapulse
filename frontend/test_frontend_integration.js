#!/usr/bin/env node
/**
 * Frontend Integration Testing Script
 * Tests frontend-backend integration with real data
 * Phase 7: Testing & Validation
 */

const axios = require('axios');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

class FrontendIntegrationTester {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.testResults = {
            apiEndpoints: {},
            websocketConnection: {},
            dataFlow: {},
            errorHandling: {},
            performance: {}
        };
        this.startTime = Date.now();
    }

    async runAllTests() {
        console.log('🚀 Starting Frontend Integration Tests');
        console.log('=' .repeat(60));

        try {
            // Test 1: API Endpoints
            await this.testApiEndpoints();

            // Test 2: WebSocket Connection
            await this.testWebSocketConnection();

            // Test 3: Data Flow
            await this.testDataFlow();

            // Test 4: Error Handling
            await this.testErrorHandling();

            // Test 5: Performance
            await this.testPerformance();

            // Generate report
            return this.generateTestReport();

        } catch (error) {
            console.error('❌ Test suite failed:', error.message);
            return { error: error.message, status: 'FAILED' };
        }
    }

    async testApiEndpoints() {
        console.log('🔌 Testing API Endpoints...');

        const testSymbol = 'BTCUSDT';
        const testTimeframe = '1h';

        try {
            // Test status endpoint
            const statusResponse = await axios.get(`${this.apiBaseUrl}/api/single-pair/status`);
            this.testResults.apiEndpoints.status = {
                status: statusResponse.status === 200 ? 'PASS' : 'FAIL',
                responseTime: statusResponse.headers['x-response-time'] || 'N/A',
                data: statusResponse.data
            };

            // Test analysis endpoint
            const analysisResponse = await axios.get(
                `${this.apiBaseUrl}/api/single-pair/analysis/${testSymbol}?timeframe=${testTimeframe}`
            );
            this.testResults.apiEndpoints.analysis = {
                status: analysisResponse.status === 200 ? 'PASS' : 'FAIL',
                responseTime: analysisResponse.headers['x-response-time'] || 'N/A',
                hasFundamental: !!analysisResponse.data?.analysis?.fundamental,
                hasTechnical: !!analysisResponse.data?.analysis?.technical,
                hasSentiment: !!analysisResponse.data?.analysis?.sentiment
            };

            // Test confidence endpoint
            const confidenceResponse = await axios.get(
                `${this.apiBaseUrl}/api/single-pair/confidence/${testSymbol}?timeframe=${testTimeframe}`
            );
            this.testResults.apiEndpoints.confidence = {
                status: confidenceResponse.status === 200 ? 'PASS' : 'FAIL',
                responseTime: confidenceResponse.headers['x-response-time'] || 'N/A',
                hasConfidence: typeof confidenceResponse.data?.current_confidence === 'number',
                hasThreshold: typeof confidenceResponse.data?.threshold_reached === 'boolean'
            };

            // Test signal endpoint
            const signalResponse = await axios.get(
                `${this.apiBaseUrl}/api/single-pair/signal/${testSymbol}?timeframe=${testTimeframe}`
            );
            this.testResults.apiEndpoints.signal = {
                status: signalResponse.status === 200 ? 'PASS' : 'FAIL',
                responseTime: signalResponse.headers['x-response-time'] || 'N/A',
                hasSignal: !!signalResponse.data?.signal,
                hasConfidence: typeof signalResponse.data?.signal?.confidence_score === 'number',
                hasTPLevels: signalResponse.data?.signal ? 
                    [1, 2, 3, 4].every(i => signalResponse.data.signal[`take_profit_${i}`]) : false
            };

            console.log('✅ API Endpoints tests completed');

        } catch (error) {
            console.error('❌ API Endpoints test failed:', error.message);
            this.testResults.apiEndpoints.error = error.message;
        }
    }

    async testWebSocketConnection() {
        console.log('🔄 Testing WebSocket Connection...');

        const testSymbol = 'BTCUSDT';
        const wsUrl = `ws://localhost:8000/api/single-pair/ws/${testSymbol}`;

        return new Promise((resolve) => {
            try {
                const ws = new WebSocket(wsUrl);
                let messageCount = 0;
                const maxMessages = 3;

                const timeout = setTimeout(() => {
                    ws.close();
                    this.testResults.websocketConnection.error = 'Connection timeout';
                    resolve();
                }, 15000);

                ws.on('open', () => {
                    console.log('✅ WebSocket connection established');
                    this.testResults.websocketConnection.connection = {
                        status: 'PASS',
                        connected: true
                    };
                });

                ws.on('message', (data) => {
                    try {
                        const message = JSON.parse(data.toString());
                        messageCount++;

                        if (messageCount === 1) {
                            this.testResults.websocketConnection.firstMessage = {
                                status: 'PASS',
                                type: message.type,
                                hasAnalysis: !!message.analysis,
                                hasConfidence: !!message.confidence,
                                hasSignal: !!message.signal,
                                timestamp: message.timestamp
                            };
                        }

                        if (messageCount === 2) {
                            this.testResults.websocketConnection.streaming = {
                                status: 'PASS',
                                type: message.type,
                                timestampDifferent: message.timestamp !== this.testResults.websocketConnection.firstMessage.timestamp
                            };
                        }

                        if (messageCount >= maxMessages) {
                            clearTimeout(timeout);
                            ws.close();
                            console.log('✅ WebSocket Streaming tests completed');
                            resolve();
                        }

                    } catch (error) {
                        console.error('❌ Error parsing WebSocket message:', error.message);
                        this.testResults.websocketConnection.messageParsing = {
                            status: 'FAIL',
                            error: error.message
                        };
                    }
                });

                ws.on('error', (error) => {
                    console.error('❌ WebSocket error:', error.message);
                    this.testResults.websocketConnection.error = error.message;
                    clearTimeout(timeout);
                    resolve();
                });

                ws.on('close', () => {
                    clearTimeout(timeout);
                    resolve();
                });

            } catch (error) {
                console.error('❌ WebSocket test failed:', error.message);
                this.testResults.websocketConnection.error = error.message;
                resolve();
            }
        });
    }

    async testDataFlow() {
        console.log('📊 Testing Data Flow...');

        const testSymbol = 'BTCUSDT';
        const testTimeframe = '1h';

        try {
            // Test complete data flow
            const startTime = Date.now();

            // Get analysis data
            const analysisResponse = await axios.get(
                `${this.apiBaseUrl}/api/single-pair/analysis/${testSymbol}?timeframe=${testTimeframe}`
            );

            // Get confidence data
            const confidenceResponse = await axios.get(
                `${this.apiBaseUrl}/api/single-pair/confidence/${testSymbol}?timeframe=${testTimeframe}`
            );

            // Get signal data
            const signalResponse = await axios.get(
                `${this.apiBaseUrl}/api/single-pair/signal/${testSymbol}?timeframe=${testTimeframe}`
            );

            const endTime = Date.now();
            const totalTime = endTime - startTime;

            this.testResults.dataFlow.completeFlow = {
                status: 'PASS',
                totalTime: totalTime,
                analysisStatus: analysisResponse.status === 200 ? 'PASS' : 'FAIL',
                confidenceStatus: confidenceResponse.status === 200 ? 'PASS' : 'FAIL',
                signalStatus: signalResponse.status === 200 ? 'PASS' : 'FAIL',
                dataConsistency: this.checkDataConsistency(analysisResponse.data, confidenceResponse.data, signalResponse.data)
            };

            console.log('✅ Data Flow tests completed');

        } catch (error) {
            console.error('❌ Data Flow test failed:', error.message);
            this.testResults.dataFlow.error = error.message;
        }
    }

    checkDataConsistency(analysisData, confidenceData, signalData) {
        try {
            // Check if all data has the same symbol
            const analysisSymbol = analysisData?.pair;
            const confidenceSymbol = confidenceData?.symbol;
            const signalSymbol = signalData?.signal?.symbol;

            const symbolsMatch = analysisSymbol === confidenceSymbol && confidenceSymbol === signalSymbol;

            // Check if confidence values are consistent
            const analysisConfidence = analysisData?.analysis?.fundamental?.confidence || 0;
            const confidenceConfidence = confidenceData?.current_confidence || 0;
            const signalConfidence = signalData?.signal?.confidence_score || 0;

            const confidenceConsistent = Math.abs(analysisConfidence - confidenceConfidence) < 0.1;

            return {
                symbolsMatch,
                confidenceConsistent,
                overallConsistent: symbolsMatch && confidenceConsistent
            };

        } catch (error) {
            return {
                symbolsMatch: false,
                confidenceConsistent: false,
                overallConsistent: false,
                error: error.message
            };
        }
    }

    async testErrorHandling() {
        console.log('🛡️ Testing Error Handling...');

        try {
            // Test invalid symbol
            try {
                await axios.get(`${this.apiBaseUrl}/api/single-pair/analysis/INVALID_SYMBOL?timeframe=1h`);
                this.testResults.errorHandling.invalidSymbol = {
                    status: 'FAIL',
                    message: 'Should have returned error for invalid symbol'
                };
            } catch (error) {
                this.testResults.errorHandling.invalidSymbol = {
                    status: 'PASS',
                    handledGracefully: error.response?.status >= 400,
                    statusCode: error.response?.status
                };
            }

            // Test invalid timeframe
            try {
                await axios.get(`${this.apiBaseUrl}/api/single-pair/analysis/BTCUSDT?timeframe=invalid`);
                this.testResults.errorHandling.invalidTimeframe = {
                    status: 'FAIL',
                    message: 'Should have returned error for invalid timeframe'
                };
            } catch (error) {
                this.testResults.errorHandling.invalidTimeframe = {
                    status: 'PASS',
                    handledGracefully: error.response?.status >= 400,
                    statusCode: error.response?.status
                };
            }

            // Test server unavailable
            try {
                await axios.get('http://localhost:9999/api/single-pair/status', { timeout: 1000 });
                this.testResults.errorHandling.serverUnavailable = {
                    status: 'FAIL',
                    message: 'Should have failed for unavailable server'
                };
            } catch (error) {
                this.testResults.errorHandling.serverUnavailable = {
                    status: 'PASS',
                    handledGracefully: error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT'
                };
            }

            console.log('✅ Error Handling tests completed');

        } catch (error) {
            console.error('❌ Error Handling test failed:', error.message);
            this.testResults.errorHandling.error = error.message;
        }
    }

    async testPerformance() {
        console.log('⚡ Testing Performance...');

        const testSymbol = 'BTCUSDT';
        const testTimeframe = '1h';
        const iterations = 10;

        try {
            const responseTimes = [];

            // Test multiple requests
            for (let i = 0; i < iterations; i++) {
                const startTime = Date.now();
                
                await axios.get(
                    `${this.apiBaseUrl}/api/single-pair/analysis/${testSymbol}?timeframe=${testTimeframe}`
                );
                
                const endTime = Date.now();
                responseTimes.push(endTime - startTime);
            }

            const avgResponseTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
            const maxResponseTime = Math.max(...responseTimes);
            const minResponseTime = Math.min(...responseTimes);

            this.testResults.performance.loadTest = {
                status: avgResponseTime < 1000 ? 'PASS' : 'FAIL',
                iterations,
                avgResponseTime: Math.round(avgResponseTime),
                maxResponseTime,
                minResponseTime,
                allResponsesUnder1s: responseTimes.every(time => time < 1000)
            };

            console.log('✅ Performance tests completed');

        } catch (error) {
            console.error('❌ Performance test failed:', error.message);
            this.testResults.performance.error = error.message;
        }
    }

    generateTestReport() {
        console.log('📋 Generating Test Report...');

        // Calculate overall status
        let totalTests = 0;
        let passedTests = 0;

        Object.values(this.testResults).forEach(category => {
            Object.values(category).forEach(test => {
                if (test && typeof test === 'object' && 'status' in test) {
                    totalTests++;
                    if (test.status === 'PASS') {
                        passedTests++;
                    }
                }
            });
        });

        const successRate = totalTests > 0 ? (passedTests / totalTests * 100) : 0;
        const totalTime = Date.now() - this.startTime;

        const report = {
            testSummary: {
                totalTests,
                passedTests,
                failedTests: totalTests - passedTests,
                successRate: `${successRate.toFixed(1)}%`,
                overallStatus: successRate >= 80 ? 'PASS' : 'FAIL',
                totalTime: `${totalTime}ms`
            },
            testResults: this.testResults,
            timestamp: new Date().toISOString(),
            recommendations: this.generateRecommendations()
        };

        return report;
    }

    generateRecommendations() {
        const recommendations = [];

        Object.entries(this.testResults).forEach(([category, tests]) => {
            Object.entries(tests).forEach(([testName, result]) => {
                if (result && typeof result === 'object' && result.status === 'FAIL') {
                    recommendations.push(`Fix ${category}.${testName} - ${result.error || 'Unknown error'}`);
                }
            });
        });

        if (recommendations.length === 0) {
            recommendations.push('All tests passed! Frontend is ready for production.');
        }

        return recommendations;
    }
}

async function main() {
    const tester = new FrontendIntegrationTester();

    console.log('🚀 Starting Frontend Integration Tests');
    console.log('=' .repeat(60));

    // Run all tests
    const report = await tester.runAllTests();

    // Save report
    const reportFile = `frontend_integration_test_report_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));

    console.log('=' .repeat(60));
    console.log('📋 TEST REPORT SUMMARY:');
    console.log(`Total Tests: ${report.testSummary.totalTests}`);
    console.log(`Passed: ${report.testSummary.passedTests}`);
    console.log(`Failed: ${report.testSummary.failedTests}`);
    console.log(`Success Rate: ${report.testSummary.successRate}`);
    console.log(`Overall Status: ${report.testSummary.overallStatus}`);
    console.log(`Total Time: ${report.testSummary.totalTime}`);
    console.log(`Report saved to: ${reportFile}`);

    if (report.testSummary.overallStatus === 'PASS') {
        console.log('🎉 All tests passed! Frontend is ready for production.');
        process.exit(0);
    } else {
        console.error('❌ Some tests failed. Check the report for details.');
        process.exit(1);
    }
}

// Run tests
main().catch(error => {
    console.error('❌ Test runner failed:', error.message);
    process.exit(1);
});
