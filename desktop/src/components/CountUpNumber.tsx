import React, { useEffect, useState, useRef } from 'react';

interface CountUpNumberProps {
  targetValue: number;
  duration?: number;
  decimals?: number;
  suffix?: string;
  prefix?: string;
  className?: string;
}

export const CountUpNumber: React.FC<CountUpNumberProps> = ({
  targetValue,
  duration = 900,
  decimals = 1,
  suffix = '',
  prefix = '',
  className = '',
}) => {
  const [displayValue, setDisplayValue] = useState<number>(0);
  const frameRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);

  useEffect(() => {
    // Easing function: easeOutCubic
    const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);

    const startVal = 0;
    const endVal = targetValue;
    startTimeRef.current = null;

    const animate = (timestamp: number) => {
      if (startTimeRef.current === null) {
        startTimeRef.current = timestamp;
      }
      const elapsed = timestamp - startTimeRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const easedProgress = easeOutCubic(progress);
      const current = startVal + (endVal - startVal) * easedProgress;

      setDisplayValue(current);

      if (progress < 1) {
        frameRef.current = requestAnimationFrame(animate);
      } else {
        setDisplayValue(endVal);
      }
    };

    frameRef.current = requestAnimationFrame(animate);

    return () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, [targetValue, duration]);

  return (
    <span className={className}>
      {prefix}
      {displayValue.toFixed(decimals)}
      {suffix}
    </span>
  );
};
