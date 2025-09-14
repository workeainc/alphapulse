-- Migration: Enhance signals table with Deep Learning
-- This migration adds support for storing DL data in the signals table

-- Add new columns for Deep Learning
ALTER TABLE enhanced_signals 
ADD COLUMN IF NOT EXISTS dl_analysis JSONB,
ADD COLUMN IF NOT EXISTS lstm_prediction DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS cnn_prediction DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS lstm_cnn_prediction DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS ensemble_prediction DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS dl_confidence DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS dl_bias VARCHAR(10);  -- 'bullish', 'bearish', 'neutral'

-- Create indexes for DL queries
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_dl_confidence ON enhanced_signals(dl_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_dl_bias ON enhanced_signals(dl_bias);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_ensemble_prediction ON enhanced_signals(ensemble_prediction DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_dl_analysis ON enhanced_signals USING GIN (dl_analysis);

-- Create a view for AI-enhanced high-quality signals
CREATE OR REPLACE VIEW ai_enhanced_signals AS
SELECT 
    id,
    symbol,
    side,
    strategy,
    confidence,
    strength,
    timestamp,
    price,
    stop_loss,
    take_profit,
    signal_quality_score,
    confirmation_count,
    smc_confidence,
    smc_bias,
    dl_confidence,
    dl_bias,
    ensemble_prediction,
    metadata,
    ichimoku_data,
    fibonacci_data,
    volume_analysis,
    advanced_indicators,
    smc_analysis,
    dl_analysis
FROM enhanced_signals
WHERE confidence >= 0.7 
  AND signal_quality_score >= 0.6
  AND confirmation_count >= 2
  AND smc_confidence >= 0.6
  AND dl_confidence >= 0.6
ORDER BY (confidence + smc_confidence + dl_confidence) / 3 DESC, timestamp DESC;

-- Create a function to calculate AI-enhanced signal quality
CREATE OR REPLACE FUNCTION calculate_ai_enhanced_quality(
    p_confidence DECIMAL,
    p_smc_confidence DECIMAL,
    p_dl_confidence DECIMAL,
    p_ichimoku_data JSONB,
    p_fibonacci_data JSONB,
    p_volume_analysis JSONB,
    p_smc_analysis JSONB,
    p_dl_analysis JSONB
) RETURNS DECIMAL AS $$
DECLARE
    quality_score DECIMAL := p_confidence;
    confirmation_bonus DECIMAL := 0.0;
    smc_bonus DECIMAL := 0.0;
    dl_bonus DECIMAL := 0.0;
BEGIN
    -- Technical indicator confirmation bonus
    IF p_ichimoku_data IS NOT NULL THEN
        IF p_ichimoku_data->>'cloud_bullish' = 'true' OR p_ichimoku_data->>'cloud_bullish' = 'false' THEN
            confirmation_bonus := confirmation_bonus + 0.1;
        END IF;
        IF p_ichimoku_data->>'tenkan_kijun' IS NOT NULL THEN
            confirmation_bonus := confirmation_bonus + 0.05;
        END IF;
    END IF;
    
    IF p_fibonacci_data IS NOT NULL THEN
        IF p_fibonacci_data->>'fib_bullish' = 'true' OR p_fibonacci_data->>'fib_bullish' = 'false' THEN
            confirmation_bonus := confirmation_bonus + 0.1;
        END IF;
    END IF;
    
    IF p_volume_analysis IS NOT NULL THEN
        IF p_volume_analysis->>'volume_confirmation' = 'true' THEN
            confirmation_bonus := confirmation_bonus + 0.05;
        END IF;
    END IF;
    
    -- SMC bonus
    IF p_smc_confidence IS NOT NULL AND p_smc_confidence > 0.6 THEN
        smc_bonus := p_smc_confidence * 0.15;  -- SMC can add up to 15% bonus
    END IF;
    
    IF p_smc_analysis IS NOT NULL THEN
        -- Order blocks bonus
        IF (p_smc_analysis->>'order_blocks_count')::int > 0 THEN
            smc_bonus := smc_bonus + 0.05;
        END IF;
        
        -- Fair value gaps bonus
        IF (p_smc_analysis->>'fair_value_gaps_count')::int > 0 THEN
            smc_bonus := smc_bonus + 0.05;
        END IF;
        
        -- Liquidity sweeps bonus
        IF (p_smc_analysis->>'liquidity_sweeps_count')::int > 0 THEN
            smc_bonus := smc_bonus + 0.05;
        END IF;
        
        -- Market structures bonus
        IF (p_smc_analysis->>'market_structures_count')::int > 0 THEN
            smc_bonus := smc_bonus + 0.05;
        END IF;
    END IF;
    
    -- Deep Learning bonus
    IF p_dl_confidence IS NOT NULL AND p_dl_confidence > 0.6 THEN
        dl_bonus := p_dl_confidence * 0.15;  -- DL can add up to 15% bonus
    END IF;
    
    IF p_dl_analysis IS NOT NULL THEN
        -- High confidence signal bonus
        IF p_dl_analysis->>'high_confidence_signal' = 'true' THEN
            dl_bonus := dl_bonus + 0.05;
        END IF;
        
        -- Strong signal bonus
        IF (p_dl_analysis->>'signal_strength')::decimal > 0.6 THEN
            dl_bonus := dl_bonus + 0.05;
        END IF;
        
        -- Multiple models bonus
        IF (p_dl_analysis->>'models_used')::int >= 2 THEN
            dl_bonus := dl_bonus + 0.05;
        END IF;
    END IF;
    
    -- Calculate final quality score
    quality_score := LEAST(1.0, quality_score + confirmation_bonus + smc_bonus + dl_bonus);
    
    RETURN quality_score;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to automatically calculate AI-enhanced signal quality
CREATE OR REPLACE FUNCTION update_ai_enhanced_quality() RETURNS TRIGGER AS $$
BEGIN
    NEW.signal_quality_score := calculate_ai_enhanced_quality(
        NEW.confidence,
        NEW.smc_confidence,
        NEW.dl_confidence,
        NEW.ichimoku_data,
        NEW.fibonacci_data,
        NEW.volume_analysis,
        NEW.smc_analysis,
        NEW.dl_analysis
    );
    
    -- Calculate confirmation count including AI
    NEW.confirmation_count := 0;
    IF NEW.ichimoku_data IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    IF NEW.fibonacci_data IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    IF NEW.volume_analysis IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    IF NEW.smc_analysis IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    IF NEW.dl_analysis IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS trigger_update_ai_enhanced_quality ON enhanced_signals;
CREATE TRIGGER trigger_update_ai_enhanced_quality
    BEFORE INSERT OR UPDATE ON enhanced_signals
    FOR EACH ROW
    EXECUTE FUNCTION update_ai_enhanced_quality();

-- Add comments for documentation
COMMENT ON COLUMN enhanced_signals.dl_analysis IS 'Deep Learning analysis data in JSON format';
COMMENT ON COLUMN enhanced_signals.lstm_prediction IS 'LSTM model prediction (0.0-1.0)';
COMMENT ON COLUMN enhanced_signals.cnn_prediction IS 'CNN model prediction (0.0-1.0)';
COMMENT ON COLUMN enhanced_signals.lstm_cnn_prediction IS 'LSTM-CNN hybrid model prediction (0.0-1.0)';
COMMENT ON COLUMN enhanced_signals.ensemble_prediction IS 'Ensemble prediction from all DL models (0.0-1.0)';
COMMENT ON COLUMN enhanced_signals.dl_confidence IS 'Deep Learning confidence score (0.0-1.0)';
COMMENT ON COLUMN enhanced_signals.dl_bias IS 'Deep Learning bias (bullish/bearish/neutral)';
COMMENT ON VIEW ai_enhanced_signals IS 'View of high-quality signals with AI confirmation';
