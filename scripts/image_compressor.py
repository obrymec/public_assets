#!/usr/bin/env python3
"""
Smart Image Compression Tool - With full mode support including 32-bit images
"""

import os
import sys
import argparse
import warnings
from io import BytesIO
from PIL import Image, ImagePalette, ImageFilter, ImageChops, ImageOps
import numpy as np
from typing import Dict, Tuple, Optional, List

# Suppress NumPy 2.0 deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class ImageAnalyzer:
    """Analyze images to determine optimal compression strategy."""
    
    @staticmethod
    def analyze_image(img: Image.Image) -> Dict:
        """
        Comprehensive image analysis with full mode support.
        """
        analysis = {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'width': img.width,
            'height': img.height,
            'megapixels': (img.width * img.height) / 1_000_000,
            'has_transparency': False,
            'is_grayscale': False,
            'is_32bit': False,
            'is_16bit': False,
            'is_photographic': False,
            'is_graphic': False,
            'is_texture': False,
            'color_count': None,
            'entropy': None,
            'noise_level': None,
            'sharpness': None,
            'recommended_format': None,
            'recommended_quality': 85,
            'compression_potential': 'medium',
            'needs_conversion': False,
            'target_mode': None
        }
        
        # Check for special modes
        if img.mode == 'I':
            analysis['is_32bit'] = True
            analysis['is_grayscale'] = True
            analysis['needs_conversion'] = True
            analysis['target_mode'] = 'L'  # Convert to 8-bit grayscale
        
        elif img.mode == 'I;16':
            analysis['is_16bit'] = True
            analysis['is_grayscale'] = True
            analysis['needs_conversion'] = True
            analysis['target_mode'] = 'L'
        
        elif img.mode == 'F':
            analysis['is_32bit'] = True
            analysis['is_grayscale'] = True
            analysis['needs_conversion'] = True
            analysis['target_mode'] = 'L'
        
        # Check transparency for other modes
        elif img.mode in ['RGBA', 'LA', 'P']:
            if img.mode == 'RGBA':
                alpha = img.getchannel('A')
                if alpha.getextrema() != (255, 255):
                    analysis['has_transparency'] = True
            elif img.mode == 'LA':
                alpha = img.getchannel('A')
                if alpha.getextrema() != (255, 255):
                    analysis['has_transparency'] = True
            elif img.mode == 'P' and 'transparency' in img.info:
                analysis['has_transparency'] = True
        
        # Check if grayscale for standard modes
        if not analysis['is_grayscale']:
            if img.mode in ['L', '1', 'P']:
                analysis['is_grayscale'] = True
            elif img.mode == 'RGB':
                r, g, b = img.split()
                if r.tobytes() == g.tobytes() == b.tobytes():
                    analysis['is_grayscale'] = True
            elif img.mode == 'RGBA':
                r, g, b, a = img.split()
                if r.tobytes() == g.tobytes() == b.tobytes():
                    analysis['is_grayscale'] = True
        
        # Analyze color complexity
        analysis['color_count'] = ImageAnalyzer._count_colors(img)
        
        # Calculate image entropy
        analysis['entropy'] = ImageAnalyzer._calculate_entropy(img)
        
        # Estimate noise level
        analysis['noise_level'] = ImageAnalyzer._estimate_noise(img)
        
        # Estimate sharpness
        analysis['sharpness'] = ImageAnalyzer._estimate_sharpness(img)
        
        # Classify image type
        analysis = ImageAnalyzer._classify_image(img, analysis)
        
        # Determine optimal format and quality
        analysis = ImageAnalyzer._determine_optimal_compression(analysis)
        
        return analysis
    
    @staticmethod
    def _convert_special_mode(img: Image.Image, target_mode: str) -> Image.Image:
        """
        Convert special image modes (I, I;16, F) to standard modes.
        """
        if img.mode == 'I':
            # 32-bit signed integer to 8-bit
            img_array = np.array(img, dtype=np.int32)
            
            # Normalize to 0-255 range
            min_val = np.min(img_array)
            max_val = np.max(img_array)
            
            if max_val > min_val:
                normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(img_array, dtype=np.uint8)
            
            return Image.fromarray(normalized, mode='L')
        
        elif img.mode == 'I;16':
            # 16-bit to 8-bit
            img_array = np.array(img, dtype=np.uint16)
            
            # Scale down to 8-bit
            if np.max(img_array) > 255:
                scaled = (img_array / 256).astype(np.uint8)
            else:
                scaled = img_array.astype(np.uint8)
            
            return Image.fromarray(scaled, mode='L')
        
        elif img.mode == 'F':
            # 32-bit float to 8-bit
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize to 0-255 range
            min_val = np.min(img_array)
            max_val = np.max(img_array)
            
            if max_val > min_val:
                normalized = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(img_array, dtype=np.uint8)
            
            return Image.fromarray(normalized, mode='L')
        
        else:
            # Standard conversion
            return img.convert(target_mode)
    
    @staticmethod
    def _count_colors(img: Image.Image) -> int:
        """Count approximate number of unique colors."""
        try:
            # For special modes, convert first
            if img.mode in ['I', 'I;16', 'F']:
                temp_img = ImageAnalyzer._convert_special_mode(img, 'L')
            else:
                temp_img = img
            
            # For performance, sample the image if it's large
            if temp_img.width * temp_img.height > 1000000:
                sample_img = temp_img.resize((500, 500), Image.Resampling.LANCZOS)
            else:
                sample_img = temp_img
            
            if sample_img.mode in ['L', '1']:
                colors = sample_img.getcolors(maxcolors=257)
                return len(colors) if colors else 256
            
            elif sample_img.mode == 'P':
                return 256
            
            else:
                if sample_img.mode != 'RGB':
                    rgb_img = sample_img.convert('RGB')
                else:
                    rgb_img = sample_img
                
                pixels = list(rgb_img.getdata())
                if len(pixels) > 10000:
                    step = len(pixels) // 10000
                    sampled_pixels = pixels[::step]
                else:
                    sampled_pixels = pixels
                
                unique_colors = len(set(sampled_pixels))
                if len(sampled_pixels) < len(pixels):
                    estimated_colors = min(unique_colors * (len(pixels) // len(sampled_pixels)), 1000000)
                else:
                    estimated_colors = unique_colors
                
                return int(estimated_colors)
                
        except Exception:
            return 10000
    
    @staticmethod
    def _calculate_entropy(img: Image.Image) -> float:
        """Calculate image entropy."""
        try:
            # Convert special modes first
            if img.mode in ['I', 'I;16', 'F']:
                gray_img = ImageAnalyzer._convert_special_mode(img, 'L')
            elif img.mode != 'L':
                gray_img = img.convert('L')
            else:
                gray_img = img
            
            hist = gray_img.histogram()
            hist = [h for h in hist if h > 0]
            total_pixels = sum(hist)
            
            entropy = 0.0
            for h in hist:
                p = h / total_pixels
                entropy -= p * np.log2(p)
            
            return entropy
            
        except Exception:
            return 4.0
    
    @staticmethod
    def _estimate_noise(img: Image.Image) -> float:
        """Estimate noise level in image."""
        try:
            # Convert to grayscale
            if img.mode in ['I', 'I;16', 'F']:
                gray_img = ImageAnalyzer._convert_special_mode(img, 'L')
            elif img.mode != 'L':
                gray_img = img.convert('L')
            else:
                gray_img = img
            
            blurred = gray_img.filter(ImageFilter.GaussianBlur(radius=1))
            diff = ImageChops.difference(gray_img, blurred)
            diff_array = np.array(diff)
            
            noise_level = np.std(diff_array)
            return min(noise_level / 50.0, 1.0)
            
        except Exception:
            return 0.1
    
    @staticmethod
    def _estimate_sharpness(img: Image.Image) -> float:
        """Estimate image sharpness."""
        try:
            # Convert to grayscale
            if img.mode in ['I', 'I;16', 'F']:
                gray_img = ImageAnalyzer._convert_special_mode(img, 'L')
            elif img.mode != 'L':
                gray_img = img.convert('L')
            else:
                gray_img = img
            
            img_array = np.array(gray_img, dtype=np.float32)
            
            # Sobel operator for edge detection
            sobel_x = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
            
            sobel_y = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])
            
            padded = np.pad(img_array, 1, mode='edge')
            grad_x = np.zeros_like(img_array)
            grad_y = np.zeros_like(img_array)
            
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    region = padded[i:i+3, j:j+3]
                    grad_x[i, j] = np.sum(region * sobel_x)
                    grad_y[i, j] = np.sum(region * sobel_y)
            
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            sharpness = np.mean(gradient_magnitude)
            
            return min(sharpness / 100.0, 1.0)
            
        except Exception:
            return 0.5
    
    @staticmethod
    def _classify_image(img: Image.Image, analysis: Dict) -> Dict:
        """Classify image type based on analysis."""
        # Textures 3D are often 32-bit grayscale
        if analysis['is_32bit'] or analysis['is_16bit']:
            analysis['is_texture'] = True
            analysis['is_grayscale'] = True
        
        # Check if it's likely a photograph
        elif analysis['entropy'] > 5.0 and analysis['color_count'] > 1000:
            analysis['is_photographic'] = True
        
        # Check if it's likely a graphic/logo
        elif analysis['color_count'] <= 256 and analysis['entropy'] < 4.0:
            analysis['is_graphic'] = True
        
        # Check if it's likely a texture (based on patterns)
        elif analysis['sharpness'] > 0.7 and analysis['noise_level'] < 0.3:
            analysis['is_texture'] = True
        
        # Default based on image characteristics
        else:
            if analysis['has_transparency']:
                analysis['is_graphic'] = True
            else:
                analysis['is_photographic'] = True
        
        return analysis
    
    @staticmethod
    def _determine_optimal_compression(analysis: Dict) -> Dict:
        """Determine optimal compression settings based on analysis."""
        
        # Special handling for 32-bit/16-bit textures
        if analysis['is_32bit'] or analysis['is_16bit']:
            analysis['recommended_format'] = 'PNG'
            analysis['recommended_quality'] = 90
            analysis['compression_potential'] = 'high'
            return analysis
        
        # For images with transparency, PNG is usually best
        if analysis['has_transparency']:
            analysis['recommended_format'] = 'PNG'
            
            if analysis['is_graphic'] or analysis['color_count'] <= 256:
                analysis['recommended_quality'] = 100  # Lossless
                analysis['compression_potential'] = 'high'
            else:
                analysis['recommended_quality'] = 90
                analysis['compression_potential'] = 'medium'
        
        # For photographic images without transparency
        elif analysis['is_photographic']:
            analysis['recommended_format'] = 'JPEG'
            
            if analysis['noise_level'] > 0.5:
                analysis['recommended_quality'] = 75
                analysis['compression_potential'] = 'high'
            elif analysis['entropy'] > 6.0:
                analysis['recommended_quality'] = 85
                analysis['compression_potential'] = 'medium'
            else:
                analysis['recommended_quality'] = 80
                analysis['compression_potential'] = 'medium'
        
        # For graphics without transparency
        elif analysis['is_graphic']:
            analysis['recommended_format'] = 'PNG'
            analysis['recommended_quality'] = 100
            analysis['compression_potential'] = 'high'
        
        # For textures
        elif analysis['is_texture']:
            if analysis['is_grayscale']:
                analysis['recommended_format'] = 'PNG'
                analysis['recommended_quality'] = 90
            else:
                analysis['recommended_format'] = 'JPEG'
                analysis['recommended_quality'] = 80
            analysis['compression_potential'] = 'high'
        
        # Default fallback
        else:
            analysis['recommended_format'] = 'PNG'
            analysis['recommended_quality'] = 85
            analysis['compression_potential'] = 'medium'
        
        return analysis


class SmartCompressor:
    """Smart image compressor with full mode support."""
    
    def __init__(self, preserve_metadata: bool = True):
        self.preserve_metadata = preserve_metadata
        self.analyzer = ImageAnalyzer()
    
    def compress_image(self, input_path: str, output_path: str, 
                      target_size_kb: Optional[float] = None,
                      max_width: Optional[int] = None,
                      max_height: Optional[int] = None) -> Tuple[bool, float, float, float]:
        """
        Compress image intelligently while preserving quality.
        """
        try:
            with Image.open(input_path) as img:
                # Preserve metadata if requested
                metadata = img.info.copy() if self.preserve_metadata else {}
                
                original_size_kb = os.path.getsize(input_path) / 1024.0
                
                # Analyze the image
                analysis = self.analyzer.analyze_image(img)
                
                # Convert special modes if needed
                if analysis['needs_conversion']:
                    print(f"  🔄 Converting {img.mode} to {analysis['target_mode']}")
                    img = self.analyzer._convert_special_mode(img, analysis['target_mode'])
                
                # Resize if needed
                if max_width or max_height:
                    img = self._resize_image(img, max_width, max_height)
                
                # Apply optimal compression
                if target_size_kb:
                    success, new_size_kb = self._compress_to_target_size(
                        img, output_path, original_size_kb, target_size_kb, 
                        analysis, metadata
                    )
                else:
                    success, new_size_kb = self._compress_optimally(
                        img, output_path, analysis, metadata
                    )
                
                if not success:
                    return False, original_size_kb, 0, 0
                
                # Calculate results
                reduction = ((original_size_kb - new_size_kb) / original_size_kb) * 100
                
                # Print analysis summary
                self._print_analysis_summary(analysis, original_size_kb, new_size_kb, reduction)
                
                return True, original_size_kb, new_size_kb, reduction
                
        except Exception as e:
            print(f"Error compressing {input_path}: {str(e)}")
            return False, 0, 0, 0
    
    def _resize_image(self, img: Image.Image, max_width: Optional[int], 
                     max_height: Optional[int]) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        original_width, original_height = img.size
        
        if max_width and max_height:
            ratio = min(max_width / original_width, max_height / original_height)
        elif max_width:
            ratio = max_width / original_width
        elif max_height:
            ratio = max_height / original_height
        else:
            return img
        
        if ratio < 1:
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
    
    def _compress_optimally(self, img: Image.Image, output_path: str,
                          analysis: Dict, metadata: Dict) -> Tuple[bool, float]:
        """Apply optimal compression based on analysis."""
        format_methods = {
            'JPEG': self._compress_jpeg,
            'PNG': self._compress_png,
            'WEBP': self._compress_webp
        }
        
        recommended_format = analysis['recommended_format']
        quality = analysis['recommended_quality']
        
        if recommended_format in format_methods:
            success = format_methods[recommended_format](img, output_path, quality, metadata)
        else:
            success = self._compress_fallback(img, output_path, quality, metadata)
        
        if success:
            new_size_kb = os.path.getsize(output_path) / 1024.0
            return True, new_size_kb
        else:
            return False, 0
    
    def _compress_to_target_size(self, img: Image.Image, output_path: str,
                               original_size_kb: float, target_size_kb: float,
                               analysis: Dict, metadata: Dict) -> Tuple[bool, float]:
        """Compress image to meet target size."""
        success, current_size_kb = self._compress_optimally(img, output_path, analysis, metadata)
        
        if not success:
            return False, 0
        
        if current_size_kb <= target_size_kb:
            return True, current_size_kb
        
        # Try more aggressive compression
        formats_to_try = ['WEBP', 'JPEG', 'PNG']
        quality_levels = [85, 75, 65, 55, 45, 35]
        
        for fmt in formats_to_try:
            for quality in quality_levels:
                if fmt == analysis['recommended_format'] and quality >= analysis['recommended_quality']:
                    continue
                
                temp_path = output_path + '.tmp'
                
                if fmt == 'JPEG':
                    success = self._compress_jpeg(img, temp_path, quality, metadata)
                elif fmt == 'PNG':
                    success = self._compress_png(img, temp_path, quality, metadata)
                elif fmt == 'WEBP':
                    success = self._compress_webp(img, temp_path, quality, metadata)
                
                if success:
                    temp_size_kb = os.path.getsize(temp_path) / 1024.0
                    
                    if temp_size_kb <= target_size_kb:
                        os.replace(temp_path, output_path)
                        return True, temp_size_kb
                    else:
                        os.remove(temp_path)
        
        return True, current_size_kb
    
    def _compress_jpeg(self, img: Image.Image, output_path: str,
                      quality: int, metadata: Dict) -> bool:
        """Compress as JPEG with optimal settings."""
        try:
            # Handle special cases for JPEG conversion
            if img.mode in ['I', 'I;16', 'F']:
                # Convert 32-bit/16-bit to 8-bit grayscale
                img = self.analyzer._convert_special_mode(img, 'L')
            
            elif img.mode in ['RGBA', 'LA', 'P', 'CMYK']:
                img = img.convert('RGB')
            
            elif img.mode == 'L':
                # Grayscale is fine for JPEG
                pass
            
            elif img.mode != 'RGB':
                # Convert any other mode to RGB
                img = img.convert('RGB')
            
            # Save with optimal settings
            save_params = {
                'format': 'JPEG',
                'quality': quality,
                'optimize': True,
                'progressive': True,
            }
            
            if metadata:
                save_params.update(metadata)
            
            img.save(output_path, **save_params)
            return True
            
        except Exception as e:
            print(f"JPEG compression failed: {str(e)}")
            return False
    
    def _compress_png(self, img: Image.Image, output_path: str,
                     quality: int, metadata: Dict) -> bool:
        """Compress as PNG with optimal settings."""
        try:
            # For PNG, quality affects compression level (1-9)
            compress_level = max(1, min(9, quality // 10))
            
            # Handle special modes for PNG
            if img.mode in ['I', 'I;16', 'F']:
                # Keep as grayscale but ensure it's 8-bit
                if img.mode != 'L':
                    img = self.analyzer._convert_special_mode(img, 'L')
            
            save_params = {
                'format': 'PNG',
                'optimize': True,
                'compress_level': compress_level,
            }
            
            if metadata:
                save_params.update(metadata)
            
            img.save(output_path, **save_params)
            return True
            
        except Exception as e:
            print(f"PNG compression failed: {str(e)}")
            return False
    
    def _compress_webp(self, img: Image.Image, output_path: str,
                      quality: int, metadata: Dict) -> bool:
        """Compress as WebP with optimal settings."""
        try:
            # Handle special modes for WebP
            if img.mode in ['I', 'I;16', 'F']:
                if img.mode != 'L':
                    img = self.analyzer._convert_special_mode(img, 'L')
            
            lossless = quality >= 95
            
            save_params = {
                'format': 'WEBP',
                'quality': quality,
                'lossless': lossless,
                'method': 4,
            }
            
            if metadata:
                save_params.update(metadata)
            
            img.save(output_path, **save_params)
            return True
            
        except Exception as e:
            print(f"WebP compression failed: {str(e)}")
            return False
    
    def _compress_fallback(self, img: Image.Image, output_path: str,
                          quality: int, metadata: Dict) -> bool:
        """Fallback compression method."""
        try:
            # Try PNG as universal fallback
            save_params = {
                'format': 'PNG',
                'optimize': True,
                'compress_level': 6,
            }
            
            if metadata:
                save_params.update(metadata)
            
            img.save(output_path, **save_params)
            return True
            
        except Exception as e:
            print(f"Fallback compression failed: {str(e)}")
            return False
    
    def _print_analysis_summary(self, analysis: Dict, original_size: float,
                              new_size: float, reduction: float):
        """Print detailed analysis summary."""
        print(f"\n📊 Image Analysis:")
        
        mode_info = analysis['mode']
        if analysis['is_32bit']:
            mode_info += " (32-bit)"
        elif analysis['is_16bit']:
            mode_info += " (16-bit)"
        
        print(f"   Type: {mode_info} {analysis['width']}x{analysis['height']} "
              f"({analysis['megapixels']:.1f} MP)")
        
        if analysis['is_photographic']:
            print(f"   Classification: 📸 Photographic")
        elif analysis['is_graphic']:
            print(f"   Classification: 🎨 Graphic/Logo")
        elif analysis['is_texture']:
            print(f"   Classification: 🔳 Texture")
        
        if analysis['color_count']:
            print(f"   Colors: {analysis['color_count']:,} unique")
        
        print(f"   Complexity: {analysis['entropy']:.1f}/8 bits")
        print(f"   Sharpness: {analysis['sharpness']:.2f}/1.0")
        
        if analysis['has_transparency']:
            print(f"   Transparency: ✅ Present")
        
        if analysis['needs_conversion']:
            print(f"   Conversion: 🔄 {analysis['mode']} → {analysis['target_mode']}")
        
        print(f"\n⚙️ Optimal Compression:")
        print(f"   Format: {analysis['recommended_format']}")
        print(f"   Quality: {analysis['recommended_quality']}/100")
        print(f"   Potential: {analysis['compression_potential'].upper()}")
        
        print(f"\n📈 Results:")
        print(f"   Size: {original_size:.1f}KB → {new_size:.1f}KB")
        print(f"   Reduction: {reduction:.1f}%")
        
        if reduction < 20:
            rating = "🎯 Excellent quality preservation"
        elif reduction < 40:
            rating = "👍 Good balance"
        elif reduction < 60:
            rating = "⚠️ Moderate compression"
        else:
            rating = "💥 Aggressive compression"
        
        print(f"   Rating: {rating}")
        print(f"{'-'*40}")


# Le reste du code (process_directory_smart, main, etc.) reste identique
# à la version précédente, juste avec cette classe corrigée

def process_directory_smart(input_dir: str, output_dir: str,
                          target_size_kb: Optional[float] = None,
                          max_width: Optional[int] = None,
                          max_height: Optional[int] = None,
                          preserve_metadata: bool = True,
                          file_extensions: Optional[List[str]] = None) -> None:
    """Process directory with smart compression."""
    if file_extensions is None:
        file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tga', 
                          '.tiff', '.tif', '.webp', '.dds']
    
    file_extensions = [ext.lower() for ext in file_extensions]
    
    compressor = SmartCompressor(preserve_metadata=preserve_metadata)
    
    processed_count = 0
    error_count = 0
    total_original_size = 0
    total_new_size = 0
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in file_extensions:
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                print(f"\n{'='*60}")
                print(f"Processing: {relative_path}")
                print(f"{'='*60}")
                
                success, original_size, new_size, reduction = compressor.compress_image(
                    input_path, output_path, target_size_kb, max_width, max_height
                )
                
                if success:
                    processed_count += 1
                    total_original_size += original_size
                    total_new_size += new_size
                else:
                    error_count += 1
                    print(f"❌ Failed to compress: {relative_path}")
    
    if processed_count > 0:
        total_reduction = ((total_original_size - total_new_size) / 
                          total_original_size) * 100
        
        print(f"\n{'='*60}")
        print("🎉 SMART COMPRESSION SUMMARY")
        print(f"{'='*60}")
        print(f"📁 Files processed: {processed_count}")
        print(f"❌ Errors: {error_count}")
        print(f"💾 Total size: {total_original_size:.1f}KB → {total_new_size:.1f}KB")
        print(f"📉 Overall reduction: {total_reduction:.1f}%")
        print(f"💵 Space saved: {total_original_size - total_new_size:.1f}KB")
        
        if total_reduction < 25:
            print(f"🏆 Overall quality: EXCELLENT preservation")
        elif total_reduction < 40:
            print(f"👍 Overall quality: VERY GOOD preservation")
        elif total_reduction < 55:
            print(f"⚠️ Overall quality: GOOD balance")
        else:
            print(f"💡 Overall quality: AGGRESSIVE compression")
        
        print(f"{'='*60}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Smart Image Compression - With full mode support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input_folder output_folder
  %(prog)s input_folder output_folder --target-size 500
  %(prog)s input_folder output_folder --max-width 1920
        """
    )
    
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('output_dir', help='Output directory for compressed images')
    parser.add_argument('--target-size', type=float,
                       help='Target size per image in KB')
    parser.add_argument('--max-width', type=int,
                       help='Maximum width in pixels')
    parser.add_argument('--max-height', type=int,
                       help='Maximum height in pixels')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Strip EXIF and other metadata')
    parser.add_argument('--extensions', nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tga', 
                                '.tiff', '.webp'],
                       help='File extensions to process')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print("🚀 SMART IMAGE COMPRESSION - FULL MODE SUPPORT")
    print(f"{'='*60}")
    print(f"📂 Input: {args.input_dir}")
    print(f"📂 Output: {args.output_dir}")
    
    if args.target_size:
        print(f"🎯 Target size: {args.target_size}KB per image")
    else:
        print(f"💡 Mode: Quality-optimized compression")
    
    if args.max_width or args.max_height:
        print(f"📐 Resolution limit: {args.max_width or 'auto'}x{args.max_height or 'auto'}")
    
    print(f"📄 Formats: {', '.join(args.extensions)}")
    print(f"📋 Metadata: {'Preserved' if not args.no_metadata else 'Stripped'}")
    print(f"{'='*60}")
    
    process_directory_smart(
        args.input_dir,
        args.output_dir,
        args.target_size,
        args.max_width,
        args.max_height,
        not args.no_metadata,
        args.extensions
    )


if __name__ == "__main__":
    main()

