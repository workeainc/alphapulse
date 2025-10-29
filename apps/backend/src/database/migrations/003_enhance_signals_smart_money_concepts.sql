-- Migration: Enhance signals table with Smart Money Concepts
-- This migration adds support for storing SMC data in the signals table

-- Add new columns for Smart Money Concepts
ALTER TABLE enhanced_signals 
ADD COLUMN IF NOT EXISTS smc_analysis JSONB,
ADD COLUMN IF NOT EXISTS order_blocks_data JSONB,
ADD COLUMN IF NOT EXISTS fair_value_gaps_data JSONB,
ADD COLUMN IF NOT EXISTS liquidity_sweeps_data JSONB,
ADD COLUMN IF NOT EXISTS market_structures_data JSONB,
ADD COLUMN IF NOT EXISTS smc_confidence DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS smc_bias VARCHAR(10);  -- 'bullish', 'bearish', 'neutral'

-- Create indexes for SMC queries
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_smc_confidence ON enhanced_signals(smc_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_smc_bias ON enhanced_signals(smc_bias);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_smc_analysis ON enhanced_signals USING GIN (smc_analysis);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_order_blocks ON enhanced_signals USING GIN (order_blocks_data);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_fair_value_gaps ON enhanced_signals USING GIN (fair_value_gaps_data);

-- Create a view for SMC-enhanced high-quality signals
CREATE OR REPLACE VIEW smc_enhanced_signals AS
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
    metadata,
    ichimoku_data,
    fibonacci_data,
    volume_analysis,
    advanced_indicators,
    smc_analysis,
    order_blocks_data,
    fair_value_gaps_data,
    liquidity_sweeps_data,
    market_structures_data
FROM enhanced_signals
WHERE confidence >= 0.7 
  AND signal_quality_score >= 0.6
  AND confirmation_count >= 2
  AND smc_confidence >= 0.6
ORDER BY (confidence + smc_confidence) / 2 DESC, timestamp DESC;

-- Create a function to calculate SMC-enhanced signal quality
CREATE OR REPLACE FUNCTION calculate_smc_enhanced_quality(
    p_confidence DECIMAL,
    p_smc_confidence DECIMAL,
    p_ichimoku_data JSONB,
    p_fibonacci_data JSONB,
    p_volume_analysis JSONB,
    p_smc_analysis JSONB
) RETURNS DECIMAL AS $$
DECLARE
    quality_score DECIMAL := p_confidence;
    confirmation_bonus DECIMAL := 0.0;
    smc_bonus DECIMAL := 0.0;
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
        smc_bonus := p_smc_confidence * 0.2;  -- SMC can add up to 20% bonus
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
    
    -- Calculate final quality score
    quality_score := LEAST(1.0, quality_score + confirmation_bonus + smc_bonus);
    
    RETURN quality_score;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to automatically calculate SMC-enhanced signal quality
CREATE OR REPLACE FUNCTION update_smc_enhanced_quality() RETURNS TRIGGER AS $$
BEGIN
    NEW.signal_quality_score := calculate_smc_enhanced_quality(
        NEW.confidence,
        NEW.smc_confidence,
        NEW.ichimoku_data,
        NEW.fibonacci_data,
        NEW.volume_analysis,
        NEW.smc_analysis
    );
    
    -- Calculate confirmation count including SMC
    NEW.confirmation_count := 0;
    IF NEW.ichimoku_data IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    IF NEW.fibonacci_data IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    IF NEW.volume_analysis IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    IF NEW.smc_analysis IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS trigger_update_smc_enhanced_quality ON enhanced_signals;
CREATE TRIGGER trigger_update_smc_enhanced_quality
    BEFORE INSERT OR UPDATE ON enhanced_signals
    FOR EACH ROW
    EXECUTE FUNCTION update_smc_enhanced_quality();

-- Add comments for documentation
COMMENT ON COLUMN enhanced_signals.smc_analysis IS 'Smart Money Concepts analysis data in JSON format';
COMMENT ON COLUMN enhanced_signals.order_blocks_data IS 'Order blocks detection data in JSON format';
COMMENT ON COLUMN enhanced_signals.fair_value_gaps_data IS 'Fair value gaps detection data in JSON format';
COMMENT ON COLUMN enhanced_signals.liquidity_sweeps_data IS 'Liquidity sweeps detection data in JSON format';
COMMENT ON COLUMN enhanced_signals.market_structures_data IS 'Market structure analysis data in JSON format';
COMMENT ON COLUMN enhanced_signals.smc_confidence IS 'Smart Money Concepts confidence score (0.0-1.0)';
COMMENT ON COLUMN enhanced_signals.smc_bias IS 'Smart Money Concepts bias (bullish/bearish/neutral)';
COMMENT ON VIEW smc_enhanced_signals IS 'View of high-quality signals with SMC confirmation';
