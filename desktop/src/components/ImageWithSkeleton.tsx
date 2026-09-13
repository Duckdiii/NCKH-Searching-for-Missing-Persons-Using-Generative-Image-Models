import React, { useState, useEffect } from 'react';
import { Skeleton } from './Skeleton';
import { ImageIcon } from 'lucide-react';

interface ImageWithSkeletonProps {
  src?: string | null;
  alt?: string;
  className?: string;
  containerClassName?: string;
  onClick?: () => void;
  aspectRatio?: string;
  fallbackText?: string;
}

export const ImageWithSkeleton: React.FC<ImageWithSkeletonProps> = ({
  src,
  alt = 'Image',
  className = '',
  containerClassName = '',
  onClick,
  aspectRatio = 'aspect-square',
  fallbackText = 'Chưa có ảnh',
}) => {
  const [isLoaded, setIsLoaded] = useState<boolean>(false);
  const [hasError, setHasError] = useState<boolean>(false);

  useEffect(() => {
    setIsLoaded(false);
    setHasError(false);
  }, [src]);

  return (
    <div
      onClick={onClick}
      className={`relative overflow-hidden w-full ${aspectRatio} ${containerClassName}`}
    >
      {/* Skeleton placeholder shown while loading or when no src */}
      {(!isLoaded || !src || hasError) && (
        <div className="absolute inset-0 w-full h-full flex flex-col items-center justify-center bg-[#E5E7EB] text-[#9CA3AF]">
          {!src || hasError ? (
            <div className="flex flex-col items-center justify-center p-2 text-center">
              <ImageIcon className="w-6 h-6 mb-1 opacity-60 text-[#6B7280]" />
              <span className="text-[11px] text-[#6B7280]">{hasError ? 'Lỗi tải ảnh' : fallbackText}</span>
            </div>
          ) : (
            <Skeleton className="w-full h-full absolute inset-0" />
          )}
        </div>
      )}

      {/* Actual Image */}
      {src && !hasError && (
        <img
          src={src}
          alt={alt}
          onLoad={() => setIsLoaded(true)}
          onError={() => {
            setHasError(true);
            setIsLoaded(true);
          }}
          className={`${className} ${
            isLoaded ? 'opacity-100' : 'opacity-0'
          } transition-opacity duration-300 w-full h-full`}
        />
      )}
    </div>
  );
};
