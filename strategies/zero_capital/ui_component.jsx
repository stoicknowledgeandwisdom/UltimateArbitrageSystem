import React, { useState, useEffect } from 'react';
import { Card, Tabs, Tab, Container, Row, Col, Badge, ListGroup, Alert, Button } from 'react-bootstrap';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faExchangeAlt, faChartLine, faClock, faPercentage, faInfoCircle } from '@fortawesome/free-solid-svg-icons';

// This component demonstrates how the UI text could be integrated into a React interface
const TriangularArbitrageExplainer = ({ onStartTrading }) => {
  const [uiText, setUiText] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);

  // Simulate loading the UI text from the JSON file
  useEffect(() => {
    // In a real implementation, you would fetch this from your API or import the file
    const fetchUIText = async () => {
      try {
        // Simulating API fetch delay
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // This would be the actual fetch in a real implementation
        // const response = await fetch('/api/strategy/triangular-arbitrage/ui-text');
        // const data = await response.json();
        
        // For demo purposes, we're using the content directly
        const data = {
          "triangular_arbitrage": {
            "overview": {
              "title": "Triangular Arbitrage",
              "description": "Triangular arbitrage is a risk-free trading strategy that exploits price discrepancies between three different cryptocurrencies to generate profit without market exposure. Our system automatically identifies and executes these opportunities across markets with zero to minimal starting capital."
            },
            "how_it_works": {
              "title": "How It Works",
              "steps": [
                "Our system continuously monitors price relationships between currency triplets (e.g., BTC, ETH, USDT)",
                "When price discrepancies are found, the system calculates potential profit after all trading fees",
                "If the profit meets your minimum threshold, the opportunity is added to the execution queue",
                "The system executes three rapid trades in sequence to capture the price difference",
                "Each complete cycle returns to the starting currency, eliminating market exposure"
              ],
              "example": "Starting with BTC → Trade BTC for ETH → Trade ETH for USDT → Trade USDT back to BTC (with profit)"
            },
            "profit_calculation": {
              "title": "Profit Calculation",
              "formula": "Profit % = (Final Amount / Initial Amount) - 1",
              "explanation": "The system calculates potential profit by simulating the complete trading cycle with current market prices, accounting for all trading fees at each step. Only opportunities above your configured profit threshold are executed.",
              "factors": [
                "Trading fees (typically 0.1-0.5% per trade)",
                "Slippage (difference between expected and actual execution price)",
                "Market depth (available liquidity at each price level)",
                "Execution speed (critical for capturing fleeting opportunities)"
              ]
            },
            "timeframes": {
              "title": "Timeframes & Execution",
              "detection": "Opportunities are typically identified within milliseconds",
              "execution": "Complete trade cycles execute in 1-3 seconds (exchange-dependent)",
              "frequency": "Profitable opportunities may appear several times per day, depending on market volatility and configured settings",
              "optimization": "The system automatically prioritizes the most profitable opportunities"
            },
            "expected_returns": {
              "title": "Expected Returns",
              "profit_range": {
                "conservative": "0.3% - 1.2% per successful trade cycle",
                "aggressive": "0.5% - 2.5% per successful trade cycle (higher profit threshold, lower success rate)"
              },
              "volume_factors": "Daily profit potential depends on your configured parameters, capital allocation, and market conditions",
              "compounding": "The system can automatically reinvest profits to grow trading capital over time"
            }
          }
        };
        
        setUiText(data.triangular_arbitrage);
        setLoading(false);
      } catch (error) {
        console.error("Error loading UI text:", error);
        setLoading(false);
      }
    };

    fetchUIText();
  }, []);

  if (loading) {
    return <div className="text-center p-5"><div className="spinner-border" role="status"></div></div>;
  }

  if (!uiText) {
    return <Alert variant="danger">Failed to load strategy information</Alert>;
  }

  const renderTabContent = () => {
    switch(activeTab) {
      case 'overview':
        return (
          <div className="py-3">
            <h3>{uiText.overview.title}</h3>
            <p className="lead">{uiText.overview.description}</p>
            <Alert variant="info">
              <FontAwesomeIcon icon={faInfoCircle} className="me-2" />
              This strategy works with zero to minimal starting capital
            </Alert>
          </div>
        );
      
      case 'how_it_works':
        return (
          <div className="py-3">
            <h3><FontAwesomeIcon icon={faExchangeAlt} className="me-2" />{uiText.how_it_works.title}</h3>
            <ListGroup variant="flush" className="my-3">
              {uiText.how_it_works.steps.map((step, index) => (
                <ListGroup.Item key={index} className="d-flex">
                  <Badge bg="primary" className="me-3">{index + 1}</Badge>
                  {step}
                </ListGroup.Item>
              ))}
            </ListGroup>
            <Card bg="light" className="mt-3">
              <Card.Body>
                <Card.Title>Example Trade Cycle</Card.Title>
                <Card.Text>{uiText.how_it_works.example}</Card.Text>
              </Card.Body>
            </Card>
          </div>
        );
      
      case 'profit_calculation':
        return (
          <div className="py-3">
            <h3><FontAwesomeIcon icon={faPercentage} className="me-2" />{uiText.profit_calculation.title}</h3>
            <div className="bg-light p-3 text-center my-3">
              <code className="fs-5">{uiText.profit_calculation.formula}</code>
            </div>
            <p>{uiText.profit_calculation.explanation}</p>
            <h5 className="mt-4">Key Factors Affecting Profitability</h5>
            <ListGroup variant="flush">
              {uiText.profit_calculation.factors.map((factor, index) => (
                <ListGroup.Item key={index}>{factor}</ListGroup.Item>
              ))}
            </ListGroup>
          </div>
        );
      
      case 'timeframes':
        return (
          <div className="py-3">
            <h3><FontAwesomeIcon icon={faClock} className="me-2" />{uiText.timeframes.title}</h3>
            <Row className="mt-4">
              <Col md={6}>
                <Card className="h-100">
                  <Card.Header>Detection Speed</Card.Header>
                  <Card.Body>
                    <Card.Text>{uiText.timeframes.detection}</Card.Text>
                  </Card.Body>
                </Card>
              </Col>
              <Col md={6}>
                <Card className="h-100">
                  <Card.Header>Execution Time</Card.Header>
                  <Card.Body>
                    <Card.Text>{uiText.timeframes.execution}</Card.Text>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
            <Row className="mt-3">
              <Col md={6}>
                <Card className="h-100">
                  <Card.Header>Opportunity Frequency</Card.Header>
                  <Card.Body>
                    <Card.Text>{uiText.timeframes.frequency}</Card.Text>
                  </Card.Body>
                </Card>
              </Col>
              <Col md={6}>
                <Card className="h-100">
                  <Card.Header>Optimization</Card.Header>
                  <Card.Body>
                    <Card.Text>{uiText.timeframes.optimization}</Card.Text>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          </div>
        );
      
      case 'expected_returns':
        return (
          <div className="py-3">
            <h3><FontAwesomeIcon icon={faChartLine} className="me-2" />{uiText.expected_returns.title}</h3>
            <Card className="mb-4 mt-3">
              <Card.Header>Profit Range Per Trade</Card.Header>
              <Card.Body>
                <Row>
                  <Col md={6}>
                    <h5 className="text-success">Conservative Strategy</h5>
                    <p className="lead">{uiText.expected_returns.profit_range.conservative}</p>
                    <small className="text-muted">Lower threshold, higher success rate</small>
                  </Col>
                  <Col md={6}>
                    <h5 className="text-warning">Aggressive Strategy</h5>
                    <p className="lead">{uiText.expected_returns.profit_range.aggressive}</p>
                    <small className="text-muted">Higher threshold, lower success rate</small>
                  </Col>
                </Row>
              </Card.Body>
            </Card>
            <Alert variant="info">
              <p className="mb-1"><strong>Volume Factors:</strong> {uiText.expected_returns.volume_factors}</p>
              <p className="mb-0"><strong>Compounding:</strong> {uiText.expected_returns.compounding}</p>
            </Alert>
            <div className="text-center mt-4">
              <Button variant="primary" size="lg" onClick={onStartTrading}>
                Start Triangular Arbitrage Trading
              </Button>
            </div>
          </div>
        );
      
      default:
        return <div>Select a tab to learn more</div>;
    }
  };

  return (
    <Container className="py-4">
      <Card>
        <Card.Header>
          <Tabs
            activeKey={activeTab}
            onSelect={(tab) => setActiveTab(tab)}
            className="card-header-tabs"
          >
            <Tab eventKey="overview" title="Overview" />
            <Tab eventKey="how_it_works" title="How It Works" />
            <Tab eventKey="profit_calculation" title="Profit Calculation" />
            <Tab eventKey="timeframes" title="Timeframes" />
            <Tab eventKey="expected_returns" title="Expected Returns" />
          </Tabs>
        </Card.Header>
        <Card.Body>
          {renderTabContent()}
        </Card.Body>
      </Card>
    </Container>
  );
};

export default TriangularArbitrageExplainer;

