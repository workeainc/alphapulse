-- Migration: Enhance signals table with advanced technical indicators
-- This migration adds support for storing advanced indicator data in the signals table

-- Add new columns for advanced indicators
ALTER TABLE enhanced_signals 
ADD COLUMN IF NOT EXISTS ichimoku_data JSONB,
ADD COLUMN IF NOT EXISTS fibonacci_data JSONB,
ADD COLUMN IF NOT EXISTS volume_analysis JSONB,
ADD COLUMN IF NOT EXISTS advanced_indicators JSONB,
ADD COLUMN IF NOT EXISTS signal_quality_score DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS confirmation_count INTEGER DEFAULT 0;

-- Create indexes for advanced indicator queries
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_quality_score ON enhanced_signals(signal_quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_confirmation_count ON enhanced_signals(confirmation_count DESC);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_ichimoku ON enhanced_signals USING GIN (ichimoku_data);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_fibonacci ON enhanced_signals USING GIN (fibonacci_data);
CREATE INDEX IF NOT EXISTS idx_enhanced_signals_volume_analysis ON enhanced_signals USING GIN (volume_analysis);

-- Create a view for high-quality signals
CREATE OR REPLACE VIEW high_quality_signals AS
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
    metadata,
    ichimoku_data,
    fibonacci_data,
    volume_analysis,
    advanced_indicators
FROM enhanced_signals
WHERE confidence >= 0.7 
  AND signal_quality_score >= 0.6
  AND confirmation_count >= 2
ORDER BY timestamp DESC;

-- Create a function to calculate signal quality score
CREATE OR REPLACE FUNCTION calculate_signal_quality(
    p_confidence DECIMAL,
    p_ichimoku_data JSONB,
    p_fibonacci_data JSONB,
    p_volume_analysis JSONB
) RETURNS DECIMAL AS $$
DECLARE
    quality_score DECIMAL := p_confidence;
    confirmation_bonus DECIMAL := 0.0;
BEGIN
    -- Ichimoku confirmation bonus
    IF p_ichimoku_data IS NOT NULL THEN
        IF p_ichimoku_data->>'cloud_bullish' = 'true' OR p_ichimoku_data->>'cloud_bullish' = 'false' THEN
            confirmation_bonus := confirmation_bonus + 0.1;
        END IF;
        IF p_ichimoku_data->>'tenkan_kijun' IS NOT NULL THEN
            confirmation_bonus := confirmation_bonus + 0.05;
        END IF;
    END IF;
    
    -- Fibonacci confirmation bonus
    IF p_fibonacci_data IS NOT NULL THEN
        IF p_fibonacci_data->>'fib_bullish' = 'true' OR p_fibonacci_data->>'fib_bullish' = 'false' THEN
            confirmation_bonus := confirmation_bonus + 0.1;
        END IF;
    END IF;
    
    -- Volume confirmation bonus
    IF p_volume_analysis IS NOT NULL THEN
        IF p_volume_analysis->>'volume_confirmation' = 'true' THEN
            confirmation_bonus := confirmation_bonus + 0.05;
        END IF;
    END IF;
    
    -- Calculate final quality score
    quality_score := LEAST(1.0, quality_score + confirmation_bonus);
    
    RETURN quality_score;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to automatically calculate signal quality
CREATE OR REPLACE FUNCTION update_signal_quality() RETURNS TRIGGER AS $$
BEGIN
    NEW.signal_quality_score := calculate_signal_quality(
        NEW.confidence,
        NEW.ichimoku_data,
        NEW.fibonacci_data,
        NEW.volume_analysis
    );
    
    -- Calculate confirmation count
    NEW.confirmation_count := 0;
    IF NEW.ichimoku_data IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    IF NEW.fibonacci_data IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    IF NEW.volume_analysis IS NOT NULL THEN NEW.confirmation_count := NEW.confirmation_count + 1; END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
DROP TRIGGER IF EXISTS trigger_update_signal_quality ON enhanced_signals;
CREATE TRIGGER trigger_update_signal_quality
    BEFORE INSERT OR UPDATE ON enhanced_signals
    FOR EACH ROW
    EXECUTE FUNCTION update_signal_quality();

-- Add comments for documentation
COMMENT ON COLUMN enhanced_signals.ichimoku_data IS 'Ichimoku Cloud analysis data in JSON format';
COMMENT ON COLUMN enhanced_signals.fibonacci_data IS 'Fibonacci retracement analysis data in JSON format';
COMMENT ON COLUMN enhanced_signals.volume_analysis IS 'Volume analysis data in JSON format';
COMMENT ON COLUMN enhanced_signals.advanced_indicators IS 'All advanced technical indicators data in JSON format';
COMMENT ON COLUMN enhanced_signals.signal_quality_score IS 'Calculated signal quality score (0.0-1.0)';
COMMENT ON COLUMN enhanced_signals.confirmation_count IS 'Number of technical indicator confirmations';
COMMENT ON VIEW high_quality_signals IS 'View of high-quality signals with multiple confirmations';
