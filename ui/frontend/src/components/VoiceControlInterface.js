import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Fab,
  Chip,
  LinearProgress,
  Alert,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Divider,
  Tooltip,
  Paper,
  Slide,
  Zoom
} from '@mui/material';
import {
  Mic,
  MicOff,
  VolumeUp,
  VolumeOff,
  Psychology,
  SmartToy,
  RecordVoiceOver,
  Hearing,
  SpeakerPhone,
  Assistant,
  AutoAwesome,
  Close
} from '@mui/icons-material';

// AI Voice Commands Processor
class VoiceAI {
  constructor() {
    this.commands = {
      // System Control
      'start system': { action: 'START_SYSTEM', confidence: 0.95 },
      'launch optimization': { action: 'START_SYSTEM', confidence: 0.9 },
      'begin trading': { action: 'START_SYSTEM', confidence: 0.85 },
      'activate quantum': { action: 'START_SYSTEM', confidence: 0.8 },
      
      'stop system': { action: 'STOP_SYSTEM', confidence: 0.95 },
      'emergency stop': { action: 'EMERGENCY_STOP', confidence: 0.98 },
      'halt trading': { action: 'STOP_SYSTEM', confidence: 0.9 },
      'shutdown': { action: 'STOP_SYSTEM', confidence: 0.85 },
      
      // Risk Management
      'set risk to conservative': { action: 'SET_RISK', params: { level: 25 }, confidence: 0.9 },
      'set risk to moderate': { action: 'SET_RISK', params: { level: 50 }, confidence: 0.9 },
      'set risk to aggressive': { action: 'SET_RISK', params: { level: 75 }, confidence: 0.9 },
      'increase risk': { action: 'ADJUST_RISK', params: { delta: 10 }, confidence: 0.85 },
      'decrease risk': { action: 'ADJUST_RISK', params: { delta: -10 }, confidence: 0.85 },
      
      // Automation
      'full automation': { action: 'SET_AUTOMATION', params: { level: 100 }, confidence: 0.9 },
      'manual mode': { action: 'SET_AUTOMATION', params: { level: 0 }, confidence: 0.9 },
      'hybrid mode': { action: 'SET_AUTOMATION', params: { level: 50 }, confidence: 0.9 },
      
      // Optimization
      'optimize portfolio': { action: 'OPTIMIZE', confidence: 0.95 },
      'auto optimize': { action: 'AUTO_OPTIMIZE', confidence: 0.95 },
      'rebalance': { action: 'REBALANCE', confidence: 0.9 },
      
      // Information Queries
      'show performance': { action: 'SHOW_PERFORMANCE', confidence: 0.9 },
      'what is my return': { action: 'GET_RETURNS', confidence: 0.9 },
      'portfolio status': { action: 'GET_STATUS', confidence: 0.9 },
      'system health': { action: 'GET_HEALTH', confidence: 0.9 },
      
      // Navigation
      'show advanced view': { action: 'SHOW_ADVANCED', confidence: 0.9 },
      'simple view': { action: 'SHOW_SIMPLE', confidence: 0.9 },
      'open setup wizard': { action: 'OPEN_WIZARD', confidence: 0.9 },
      
      // Market Modes
      'enable esg mode': { action: 'TOGGLE_ESG', params: { enabled: true }, confidence: 0.9 },
      'disable esg mode': { action: 'TOGGLE_ESG', params: { enabled: false }, confidence: 0.9 },
      'turbo mode on': { action: 'TOGGLE_TURBO', params: { enabled: true }, confidence: 0.9 },
      'turbo mode off': { action: 'TOGGLE_TURBO', params: { enabled: false }, confidence: 0.9 }
    };
    
    this.responses = {
      START_SYSTEM: [
        "🚀 Activating the Ultimate Arbitrage System. Quantum engines are spinning up!",
        "⚡ System online! AI trading algorithms are now active.",
        "🎯 Portfolio optimization initiated. Let's maximize those returns!"
      ],
      STOP_SYSTEM: [
        "🛑 System shutdown complete. All positions have been safely managed.",
        "✋ Trading halted. Your portfolio is secure.",
        "🔒 System offline. Risk management protocols activated."
      ],
      EMERGENCY_STOP: [
        "🚨 Emergency stop executed! All trading activity halted immediately.",
        "⛔ Emergency protocols activated. Portfolio protection mode enabled."
      ],
      SET_RISK: [
        "⚖️ Risk tolerance updated successfully!",
        "📊 New risk parameters have been applied.",
        "🛡️ Risk management settings configured."
      ],
      OPTIMIZE: [
        "🧠 Portfolio optimization in progress...",
        "⚡ Quantum algorithms are analyzing market conditions.",
        "🎯 Finding the optimal allocation strategy."
      ],
      AUTO_OPTIMIZE: [
        "🤖 AI is analyzing current market conditions for optimal settings.",
        "🧠 Machine learning models are optimizing your configuration.",
        "⚡ Smart optimization complete! Settings updated."
      ],
      GET_STATUS: [
        "📊 Your portfolio is performing excellently!",
        "💎 All systems are operating at peak efficiency.",
        "🚀 Quantum advantage is delivering superior returns."
      ],
      DEFAULT: [
        "🤖 Command processed successfully!",
        "✅ Action completed!",
        "⚡ Done! Is there anything else I can help you with?"
      ]
    };
    
    this.isListening = false;
    this.recognition = null;
    this.synthesis = window.speechSynthesis;
  }
  
  initializeSpeechRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      throw new Error('Speech recognition not supported in this browser');
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.recognition = new SpeechRecognition();
    
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-US';
    
    return this.recognition;
  }
  
  processCommand(transcript) {
    const normalizedTranscript = transcript.toLowerCase().trim();
    
    // Find best matching command
    let bestMatch = null;
    let highestScore = 0;
    
    for (const [command, config] of Object.entries(this.commands)) {
      const score = this.calculateSimilarity(normalizedTranscript, command);
      if (score > highestScore && score > 0.6) { // Minimum confidence threshold
        highestScore = score;
        bestMatch = { command, config, score };
      }
    }
    
    if (bestMatch) {
      return {
        action: bestMatch.config.action,
        params: bestMatch.config.params || {},
        confidence: bestMatch.score * bestMatch.config.confidence,
        response: this.getRandomResponse(bestMatch.config.action)
      };
    }
    
    // If no exact match, try to extract intent using simple NLP
    return this.extractIntent(normalizedTranscript);
  }
  
  calculateSimilarity(text1, text2) {
    // Simple fuzzy matching algorithm
    const words1 = text1.split(' ');
    const words2 = text2.split(' ');
    
    let matches = 0;
    for (const word1 of words1) {
      for (const word2 of words2) {
        if (word1.includes(word2) || word2.includes(word1)) {
          matches++;
          break;
        }
      }
    }
    
    return matches / Math.max(words1.length, words2.length);
  }
  
  extractIntent(transcript) {
    // Advanced intent extraction using keyword patterns
    const intents = {
      start: ['start', 'begin', 'launch', 'activate', 'turn on', 'go'],
      stop: ['stop', 'halt', 'pause', 'end', 'turn off'],
      risk: ['risk', 'conservative', 'aggressive', 'safe', 'dangerous'],
      status: ['status', 'performance', 'how', 'what', 'show', 'display'],
      optimize: ['optimize', 'improve', 'better', 'enhance', 'tune']
    };
    
    for (const [intent, keywords] of Object.entries(intents)) {
      if (keywords.some(keyword => transcript.includes(keyword))) {
        const action = intent.toUpperCase() + '_SYSTEM';
        return {
          action,
          params: {},
          confidence: 0.7,
          response: this.getRandomResponse(action)
        };
      }
    }
    
    return {
      action: 'UNKNOWN',
      params: {},
      confidence: 0.3,
      response: "🤔 I didn't quite understand that. Try saying 'start system' or 'show performance'."
    };
  }
  
  getRandomResponse(action) {
    const responses = this.responses[action] || this.responses.DEFAULT;
    return responses[Math.floor(Math.random() * responses.length)];
  }
  
  speak(text) {
    if (this.synthesis) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1.1;
      utterance.volume = 0.8;
      
      // Use a pleasant voice if available
      const voices = this.synthesis.getVoices();
      const preferredVoice = voices.find(voice => 
        voice.name.includes('Female') || 
        voice.name.includes('Samantha') ||
        voice.name.includes('Google')
      );
      
      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }
      
      this.synthesis.speak(utterance);
    }
  }
}

const VoiceControlInterface = ({ onCommand, systemActive }) => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [commandHistory, setCommandHistory] = useState([]);
  const [showCommands, setShowCommands] = useState(false);
  const [aiResponse, setAiResponse] = useState('');
  const [confidence, setConfidence] = useState(0);
  
  const voiceAI = useRef(new VoiceAI());
  const recognition = useRef(null);
  
  useEffect(() => {
    try {
      recognition.current = voiceAI.current.initializeSpeechRecognition();
      
      recognition.current.onstart = () => {
        setIsListening(true);
        setTranscript('');
      };
      
      recognition.current.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }
        
        setTranscript(finalTranscript || interimTranscript);
        
        if (finalTranscript) {
          processVoiceCommand(finalTranscript);
        }
      };
      
      recognition.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
      
      recognition.current.onend = () => {
        setIsListening(false);
      };
      
    } catch (error) {
      console.error('Voice recognition not supported:', error);
    }
    
    return () => {
      if (recognition.current) {
        recognition.current.stop();
      }
    };
  }, []);
  
  const processVoiceCommand = async (transcript) => {
    setIsProcessing(true);
    
    try {
      const result = voiceAI.current.processCommand(transcript);
      
      setConfidence(result.confidence);
      setAiResponse(result.response);
      
      // Add to command history
      const historyItem = {
        timestamp: new Date(),
        transcript,
        action: result.action,
        confidence: result.confidence,
        response: result.response
      };
      
      setCommandHistory(prev => [historyItem, ...prev.slice(0, 9)]); // Keep last 10 commands
      
      // Execute the command
      if (result.confidence > 0.6) {
        await onCommand(result.action, result.params);
        
        // Speak the response if voice is enabled
        if (voiceEnabled) {
          voiceAI.current.speak(result.response);
        }
      } else {
        const lowConfidenceResponse = "🤔 I'm not sure I understood that correctly. Could you try rephrasing?";
        setAiResponse(lowConfidenceResponse);
        
        if (voiceEnabled) {
          voiceAI.current.speak(lowConfidenceResponse);
        }
      }
      
    } catch (error) {
      console.error('Command processing error:', error);
      setAiResponse('❌ Sorry, I encountered an error processing your command.');
    } finally {
      setIsProcessing(false);
    }
  };
  
  const toggleListening = () => {
    if (isListening) {
      recognition.current?.stop();
    } else {
      recognition.current?.start();
    }
  };
  
  const quickCommands = [
    { text: 'Start System', command: 'start system', icon: '🚀' },
    { text: 'Stop System', command: 'stop system', icon: '🛑' },
    { text: 'Optimize Portfolio', command: 'optimize portfolio', icon: '⚡' },
    { text: 'Show Performance', command: 'show performance', icon: '📊' },
    { text: 'Auto Optimize', command: 'auto optimize', icon: '🤖' },
    { text: 'System Status', command: 'system health', icon: '💎' }
  ];
  
  const executeQuickCommand = (command) => {
    processVoiceCommand(command);
  };
  
  return (
    <>
      {/* Floating Voice Control Button */}
      <Fab
        color={isListening ? 'secondary' : 'primary'}
        size="large"
        onClick={toggleListening}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          zIndex: 1000,
          background: isListening 
            ? 'linear-gradient(45deg, #ff6b6b, #ee5a24)' 
            : 'linear-gradient(45deg, #667eea, #764ba2)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
          '&:hover': {
            transform: 'scale(1.1)',
            transition: 'transform 0.2s'
          },
          animation: isListening ? 'pulse 1.5s infinite' : 'none',
          '@keyframes pulse': {
            '0%': { transform: 'scale(1)', opacity: 1 },
            '50%': { transform: 'scale(1.05)', opacity: 0.7 },
            '100%': { transform: 'scale(1)', opacity: 1 }
          }
        }}
      >
        {isListening ? <Mic sx={{ fontSize: '2rem' }} /> : <MicOff sx={{ fontSize: '2rem' }} />}
      </Fab>
      
      {/* Voice Interface Dialog */}
      <Dialog
        open={showCommands}
        onClose={() => setShowCommands(false)}
        maxWidth="md"
        fullWidth
        PaperProps={{
          sx: {
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white'
          }
        }}
      >
        <DialogTitle sx={{ textAlign: 'center', pb: 1 }}>
          <Typography variant="h4" component="div" sx={{ fontWeight: 'bold' }}>
            🎙️ Voice Command Center
          </Typography>
          <Typography variant="subtitle1" sx={{ opacity: 0.9 }}>
            Control your Ultimate Arbitrage System with natural language
          </Typography>
        </DialogTitle>
        
        <DialogContent>
          {/* Current Status */}
          <Paper elevation={3} sx={{ p: 3, mb: 3, background: 'rgba(255,255,255,0.1)' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <RecordVoiceOver sx={{ mr: 2, fontSize: '2rem' }} />
                <Typography variant="h6">
                  {isListening ? '🎤 Listening...' : '⏸️ Voice Control Ready'}
                </Typography>
              </Box>
              <IconButton 
                onClick={() => setVoiceEnabled(!voiceEnabled)}
                sx={{ color: 'white' }}
              >
                {voiceEnabled ? <VolumeUp /> : <VolumeOff />}
              </IconButton>
            </Box>
            
            {transcript && (
              <Typography variant="body1" sx={{ fontStyle: 'italic', mb: 2 }}>
                "{transcript}"
              </Typography>
            )}
            
            {isProcessing && (
              <Box sx={{ mb: 2 }}>
                <LinearProgress sx={{ mb: 1 }} />
                <Typography variant="body2">🧠 AI is processing your command...</Typography>
              </Box>
            )}
            
            {aiResponse && (
              <Alert 
                severity={confidence > 0.8 ? 'success' : confidence > 0.6 ? 'info' : 'warning'}
                sx={{ 
                  background: 'rgba(255,255,255,0.1)', 
                  color: 'white',
                  '& .MuiAlert-icon': { color: 'white' }
                }}
              >
                <Typography variant="body1">{aiResponse}</Typography>
                {confidence > 0 && (
                  <Typography variant="caption">
                    Confidence: {Math.round(confidence * 100)}%
                  </Typography>
                )}
              </Alert>
            )}
          </Paper>
          
          {/* Quick Commands */}
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <AutoAwesome sx={{ mr: 1 }} />
            Quick Commands
          </Typography>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 3 }}>
            {quickCommands.map((cmd, index) => (
              <Chip
                key={index}
                label={`${cmd.icon} ${cmd.text}`}
                onClick={() => executeQuickCommand(cmd.command)}
                sx={{
                  background: 'rgba(255,255,255,0.2)',
                  color: 'white',
                  '&:hover': {
                    background: 'rgba(255,255,255,0.3)',
                    transform: 'scale(1.05)'
                  }
                }}
              />
            ))}
          </Box>
          
          {/* Command History */}
          {commandHistory.length > 0 && (
            <>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <Hearing sx={{ mr: 1 }} />
                Recent Commands
              </Typography>
              
              <List sx={{ maxHeight: 300, overflow: 'auto' }}>
                {commandHistory.map((item, index) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemAvatar>
                        <Avatar sx={{ 
                          background: item.confidence > 0.8 ? '#4caf50' : item.confidence > 0.6 ? '#ff9800' : '#f44336'
                        }}>
                          <SmartToy />
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={item.transcript}
                        secondary={
                          <>
                            <Typography variant="caption" display="block">
                              {item.response}
                            </Typography>
                            <Typography variant="caption" color="rgba(255,255,255,0.7)">
                              {item.timestamp.toLocaleTimeString()} • {Math.round(item.confidence * 100)}% confidence
                            </Typography>
                          </>
                        }
                        sx={{ color: 'white' }}
                      />
                    </ListItem>
                    {index < commandHistory.length - 1 && <Divider sx={{ background: 'rgba(255,255,255,0.1)' }} />}
                  </React.Fragment>
                ))}
              </List>
            </>
          )}
        </DialogContent>
        
        <DialogActions sx={{ justifyContent: 'space-between', p: 3 }}>
          <Button
            onClick={toggleListening}
            variant="contained"
            size="large"
            startIcon={isListening ? <MicOff /> : <Mic />}
            sx={{
              background: isListening ? '#f44336' : '#4caf50',
              '&:hover': {
                background: isListening ? '#d32f2f' : '#388e3c'
              }
            }}
          >
            {isListening ? 'Stop Listening' : 'Start Listening'}
          </Button>
          
          <Button
            onClick={() => setShowCommands(false)}
            variant="outlined"
            sx={{ color: 'white', borderColor: 'white' }}
            startIcon={<Close />}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Voice Control Toggle */}
      <Tooltip title="Voice Commands">
        <IconButton
          onClick={() => setShowCommands(true)}
          sx={{
            position: 'fixed',
            bottom: 24,
            right: 104,
            background: 'linear-gradient(45deg, #2196f3, #21cbf3)',
            color: 'white',
            width: 56,
            height: 56,
            '&:hover': {
              background: 'linear-gradient(45deg, #1976d2, #0288d1)',
              transform: 'scale(1.1)'
            }
          }}
        >
          <Assistant sx={{ fontSize: '1.5rem' }} />
        </IconButton>
      </Tooltip>
    </>
  );
};

export default VoiceControlInterface;

