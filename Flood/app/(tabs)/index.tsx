import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, Alert, ScrollView, Linking, Image as RNImage, StyleSheet, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Paths, File } from 'expo-file-system';
import Constants from 'expo-constants';
import * as MediaLibrary from 'expo-media-library';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

// Backend base URL is injected via Expo config (extra.API_BASE)
// Set it in Flood/.env as API_BASE=...
// const API_BASE: string =
//   ((Constants.expoConfig?.extra as any)?.API_BASE as string) ??
//   'http://localhost:8000';

const API_BASE = Constants.expoConfig?.extra?.API_BASE ?? 'http://localhost:8000';

console.log('API Base URL:', API_BASE);

/**
 * Flood Impact Analysis Screen
 * 
 * Displays comprehensive flood damage metrics including:
 * - Flooded land area (blue color-coded)
 * - Vegetation loss area (green color-coded)
 * - Built structures affected (orange color-coded)
 * 
 * All metrics shown in pixels, m², and hectares when resolution is available.
 */
export default function AnalyzeScreen() {
  const insets = useSafeAreaInsets();
  const [beforeUri, setBeforeUri] = useState<string | null>(null);
  const [afterUri, setAfterUri] = useState<string | null>(null);
  const [detectedRes, setDetectedRes] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [beforeOverlayUrl, setBeforeOverlayUrl] = useState<string | null>(null);
  const [afterOverlayUrl, setAfterOverlayUrl] = useState<string | null>(null);
  const [beforeOverlayAspect, setBeforeOverlayAspect] = useState<number | null>(null);
  const [afterOverlayAspect, setAfterOverlayAspect] = useState<number | null>(null);

  useEffect(() => {
    let canceled = false;
    if (beforeOverlayUrl) {
      RNImage.getSize(
        beforeOverlayUrl,
        (width, height) => {
          if (!canceled) {
            setBeforeOverlayAspect(height ? width / height : null);
          }
        },
        () => {
          if (!canceled) {
            setBeforeOverlayAspect(null);
          }
        },
      );
    } else {
      setBeforeOverlayAspect(null);
    }
    return () => {
      canceled = true;
    };
  }, [beforeOverlayUrl]);

  useEffect(() => {
    let canceled = false;
    if (afterOverlayUrl) {
      RNImage.getSize(
        afterOverlayUrl,
        (width, height) => {
          if (!canceled) {
            setAfterOverlayAspect(height ? width / height : null);
          }
        },
        () => {
          if (!canceled) {
            setAfterOverlayAspect(null);
          }
        },
      );
    } else {
      setAfterOverlayAspect(null);
    }
    return () => {
      canceled = true;
    };
  }, [afterOverlayUrl]);

  const pickImage = useCallback(async (setter: (uri: string) => void) => {
    // Request permissions first
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Sorry, we need camera roll permissions to select images.');
      return;
    }
    
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 1,
    });
    console.log('Image picker result:', res);
    if (!res.canceled && res.assets?.[0]?.uri) {
      const uri = res.assets[0].uri;
      console.log('Selected image URI:', uri);
      setter(uri);
    }
  }, []);

  const onUpload = useCallback(async () => {
    if (!beforeUri || !afterUri) {
      Alert.alert('Select images', 'Please pick both before and after images.');
      return;
    }
    try {
      setLoading(true);
      const form = new FormData();
      
      // Extract filename and detect MIME type from extension
      const beforeName = beforeUri.split('/').pop() || 'before.png';
      const afterName = afterUri.split('/').pop() || 'after.png';
      
      const getMimeType = (filename: string) => {
        const ext = filename.split('.').pop()?.toLowerCase();
        const mimeTypes: Record<string, string> = {
          'jpg': 'image/jpeg',
          'jpeg': 'image/jpeg',
          'png': 'image/png',
          'gif': 'image/gif',
          'webp': 'image/webp',
        };
        return mimeTypes[ext || 'png'] || 'image/jpeg';
      };
      
      // Build FormData: native vs web have different requirements
      if (Platform.OS === 'web') {
        // On web, ImagePicker returns blob: URLs. Convert them to File objects.
        const beforeBlob = await fetch(beforeUri).then((r) => r.blob());
        const afterBlob = await fetch(afterUri).then((r) => r.blob());
        // Append blobs directly with filenames; avoids type conflicts with expo-file-system File
        form.append('before', beforeBlob, beforeName);
        form.append('after', afterBlob, afterName);
      } else {
        // React Native requires { uri, name, type }
        // @ts-ignore - React Native FormData accepts this format
        form.append('before', {
          uri: beforeUri,
          type: getMimeType(beforeName),
          name: beforeName,
        });
        // @ts-ignore - React Native FormData accepts this format
        form.append('after', {
          uri: afterUri,
          type: getMimeType(afterName),
          name: afterName,
        });
      }
      
      const url = `${API_BASE}/analyze/`;
      console.log('Uploading to:', url);
      console.log('Before URI:', beforeUri);
      console.log('After URI:', afterUri);
      
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 
          'Accept': 'application/json',
          // Don't set Content-Type - let the browser/RN set it automatically with boundary
        },
        body: form,
      });
      if (!resp.ok) {
        const txt = await resp.text();
        console.error('Upload failed:', resp.status, txt);
        throw new Error(txt || `Upload failed: ${resp.status}`);
      }
      const data = await resp.json();
      console.log('Upload successful:', data);
      setResult(data);
      if (typeof data?.metrics?.pixel_resolution_m === 'number') {
        setDetectedRes(data.metrics.pixel_resolution_m);
      }
      // Use the *_url fields returned by the server (already formatted correctly)
      if (data?.overlay_info) {
        setBeforeOverlayUrl(data.overlay_info.before_overlay_url ? `${API_BASE}${data.overlay_info.before_overlay_url}` : null);
        setAfterOverlayUrl(data.overlay_info.after_overlay_url ? `${API_BASE}${data.overlay_info.after_overlay_url}` : null);
      } else {
        setBeforeOverlayUrl(null);
        setAfterOverlayUrl(null);
      }
    } catch (err: any) {
      console.error(err);
      Alert.alert('Error', err?.message ?? 'Upload failed');
    } finally {
      setLoading(false);
    }
  }, [beforeUri, afterUri]);

  const onOpenGeoJSON = useCallback(() => {
    if (!result?.geojson_url) return;
    const url = `${API_BASE}${result.geojson_url}`;
    Linking.openURL(url);
  }, [result]);

  const onOpenCSV = useCallback(() => {
    if (!result?.csv_url) return;
    const url = `${API_BASE}${result.csv_url}`;
    Linking.openURL(url);
  }, [result]);

  const downloadImage = useCallback(async (imageUrl: string, filename: string) => {
    try {
      if (Platform.OS === 'web') {
        // Web: fetch, trim white borders via canvas, then download
        const img = new Image();
        img.crossOrigin = 'anonymous';
        const blob = await fetch(imageUrl).then((r) => r.blob());
        const blobUrl = URL.createObjectURL(blob);
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = (e) => reject(e);
          img.src = blobUrl;
        });

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          // Fallback: open original
          window.open(imageUrl, '_blank');
          return;
        }
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const { data, width, height } = imageData;

        const isWhite = (r: number, g: number, b: number, a: number) => a === 255 && r === 255 && g === 255 && b === 255;
        let top = 0, left = 0, right = width - 1, bottom = height - 1;

        // Find top
        while (top < height) {
          let rowWhite = true;
          for (let x = 0; x < width; x++) {
            const i = (top * width + x) * 4;
            if (!isWhite(data[i], data[i + 1], data[i + 2], data[i + 3])) { rowWhite = false; break; }
          }
          if (!rowWhite) break; top++;
        }
        // Find bottom
        while (bottom >= 0) {
          let rowWhite = true;
          for (let x = 0; x < width; x++) {
            const i = (bottom * width + x) * 4;
            if (!isWhite(data[i], data[i + 1], data[i + 2], data[i + 3])) { rowWhite = false; break; }
          }
          if (!rowWhite) break; bottom--;
        }
        // Find left
        while (left < width) {
          let colWhite = true;
          for (let y = top; y <= bottom; y++) {
            const i = (y * width + left) * 4;
            if (!isWhite(data[i], data[i + 1], data[i + 2], data[i + 3])) { colWhite = false; break; }
          }
          if (!colWhite) break; left++;
        }
        // Find right
        while (right >= 0) {
          let colWhite = true;
          for (let y = top; y <= bottom; y++) {
            const i = (y * width + right) * 4;
            if (!isWhite(data[i], data[i + 1], data[i + 2], data[i + 3])) { colWhite = false; break; }
          }
          if (!colWhite) break; right--;
        }

        const cropW = Math.max(0, right - left + 1);
        const cropH = Math.max(0, bottom - top + 1);
        if (cropW > 0 && cropH > 0 && (left > 0 || top > 0 || right < width - 1 || bottom < height - 1)) {
          const out = document.createElement('canvas');
          out.width = cropW; out.height = cropH;
          const octx = out.getContext('2d')!;
          octx.drawImage(img, left, top, cropW, cropH, 0, 0, cropW, cropH);
          out.toBlob((outBlob) => {
            if (!outBlob) {
              window.open(imageUrl, '_blank');
              return;
            }
            const a = document.createElement('a');
            a.href = URL.createObjectURL(outBlob);
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
          }, 'image/png');
        } else {
          // No white borders detected; download original
          const a = document.createElement('a');
          a.href = blobUrl; a.download = filename;
          document.body.appendChild(a);
          a.click();
          a.remove();
        }
        return;
      }

      // Request write-only access for photo library; avoid audio permission on Android 13+
      const { status } = await MediaLibrary.requestPermissionsAsync(true, ['photo']);
      if (status !== 'granted') {
        Alert.alert('Permission needed', 'Sorry, we need media library permissions to save images.');
        return;
      }

      // Show downloading message
      Alert.alert('Downloading...', 'Please wait while we save the image.');

      // Download the file using fetch and save with new API
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error('Download failed');
      }
      
      const blob = await response.blob();
      const file = new File(Paths.cache, filename);
      
      // Write the blob to the file
      const arrayBuffer = await blob.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      const writer = file.writableStream().getWriter();
      await writer.write(uint8Array);
      await writer.close();

      // Save to media library
      const asset = await MediaLibrary.createAssetAsync(file.uri);
      await MediaLibrary.createAlbumAsync('Flood Analysis', asset, false);
      
      Alert.alert('Success!', `Image saved to your gallery in "Flood Analysis" album.`);
    } catch (err: any) {
      console.error('Download error:', err);
      Alert.alert('Download failed', err?.message ?? 'Could not save image');
    }
  }, []);

  const onDownloadBeforeOverlay = useCallback(() => {
    if (!beforeOverlayUrl) return;
    downloadImage(beforeOverlayUrl, 'flood_before_overlay.png');
  }, [beforeOverlayUrl, downloadImage]);

  const onDownloadAfterOverlay = useCallback(() => {
    if (!afterOverlayUrl) return;
    downloadImage(afterOverlayUrl, 'flood_after_overlay.png');
  }, [afterOverlayUrl, downloadImage]);

  const beforeOverlayStyle = beforeOverlayAspect
    ? [overlayStyles.overlayImage, { aspectRatio: beforeOverlayAspect }]
    : [overlayStyles.overlayImage, overlayStyles.overlayFallback];

  const afterOverlayStyle = afterOverlayAspect
    ? [overlayStyles.overlayImage, { aspectRatio: afterOverlayAspect }]
    : [overlayStyles.overlayImage, overlayStyles.overlayFallback];

  const metrics = result?.metrics ?? {};
  const pixelAnalysis = result?.pixel_analysis ?? {};
  
  // Helpers
  const isNum = (v: any) => typeof v === 'number' && isFinite(v);
  const fmt = (v: any, digits = 2) =>
    isNum(v)
      ? Number(v).toLocaleString(undefined, {
          minimumFractionDigits: digits,
          maximumFractionDigits: digits,
        })
      : '—';
  const pick = (backend?: number | null, fallback?: number | null) =>
    isNum(backend) ? (backend as number) : isNum(fallback) ? (fallback as number) : null;
  
  // Compute areas from pixel counts using detected pixel resolution when available
  const quantify = useMemo(() => {
    const res = typeof metrics?.pixel_resolution_m === 'number' && isFinite(metrics.pixel_resolution_m)
      ? metrics.pixel_resolution_m as number
      : null;
    if (!res) return null;
    const r2 = res * res;

    const toNum = (v: any) => {
      const n = typeof v === 'number' ? v : Number(v);
      return Number.isFinite(n) ? n : null;
    };

    // Prefer backend pixel_analysis values when available
    const floodedPixels = toNum(
      metrics?.total_flooded_pixels ?? metrics?.new_water_pixels ?? pixelAnalysis?.water_candidate_pixels
    );
    const vegetationPixels = toNum(
      metrics?.vegetation_loss_pixels ?? pixelAnalysis?.veg_loss_pixels
    );
    const builtPixels = toNum(metrics?.built_structures_affected_pixels);

    const calc = (px: number | null) => {
      if (!px || px <= 0) return null;
      const m2 = px * r2;
      const ha = m2 / 10000;
      return { m2, ha };
    };

    return {
      flooded: calc(floodedPixels),
      vegetation: calc(vegetationPixels),
      built: calc(builtPixels),
      resolution_m: res,
    } as const;
  }, [metrics]);

  // Prefer backend-provided areas, fall back to client-computed areas
  const areas = useMemo(() => {
    const flooded_m2_backend = (metrics?.total_flooded_area_m2 ?? metrics?.new_water_area_m2) as number | undefined;
    const flooded_ha_backend = (metrics?.total_flooded_area_ha ?? metrics?.new_water_area_ha) as number | undefined;
    const veg_m2_backend = metrics?.vegetation_loss_area_m2 as number | undefined;
    const veg_ha_backend = metrics?.vegetation_loss_area_ha as number | undefined;
    const built_m2_backend = metrics?.built_structures_affected_area_m2 as number | undefined;
    const built_ha_backend = metrics?.built_structures_affected_area_ha as number | undefined;

    return {
      flooded_m2: pick(flooded_m2_backend, quantify?.flooded?.m2 ?? null),
      flooded_ha: pick(flooded_ha_backend, quantify?.flooded?.ha ?? null),
      veg_m2: pick(veg_m2_backend, quantify?.vegetation?.m2 ?? null),
      veg_ha: pick(veg_ha_backend, quantify?.vegetation?.ha ?? null),
      built_m2: pick(built_m2_backend, quantify?.built?.m2 ?? null),
      built_ha: pick(built_ha_backend, quantify?.built?.ha ?? null),
    } as const;
  }, [metrics, quantify]);

  return (
    <ScrollView 
      className="flex-1 bg-gradient-to-b from-blue-50 to-white"
      contentContainerStyle={{ paddingTop: insets.top + 16, paddingBottom: insets.bottom + 24, paddingHorizontal: 20 }}
    >
      {/* Header Section */}
      <View className="mb-6">
        <Text className="text-3xl font-bold text-gray-900 text-center mb-2">
          🌊 Flood Impact Analysis
        </Text>
        <Text className="text-base text-gray-600 text-center leading-5">
          Upload before and after images to analyze flood damage
        </Text>
      </View>

      {/* Image Selection Card */}
      <View className="bg-white rounded-2xl shadow-lg p-5 mb-5">
        <Text className="text-lg font-semibold text-gray-800 mb-4">Select Images</Text>
        
        <View className="flex-row gap-3 mb-4">
          <TouchableOpacity 
            className="flex-1 bg-blue-500 rounded-xl py-3.5 px-4 shadow-md active:bg-blue-600"
            onPress={() => pickImage((u) => setBeforeUri(u))}
          >
            <Text className="text-white font-semibold text-center text-base">📷 Before</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            className="flex-1 bg-blue-500 rounded-xl py-3.5 px-4 shadow-md active:bg-blue-600"
            onPress={() => pickImage((u) => setAfterUri(u))}
          >
            <Text className="text-white font-semibold text-center text-base">📷 After</Text>
          </TouchableOpacity>
        </View>

        <View className="flex-row gap-3">
          {beforeUri ? (
            <View className="flex-1">
              <RNImage 
                source={{ uri: beforeUri }} 
                style={{ width: '100%', height: 160, borderRadius: 12 }}
                resizeMode="cover"
              />
              <View className="absolute top-2 left-2 bg-black/60 rounded-lg px-2 py-1">
                <Text className="text-white text-xs font-semibold">Before</Text>
              </View>
            </View>
          ) : (
            <View className="flex-1 h-40 rounded-xl bg-gray-100 items-center justify-center border-2 border-dashed border-gray-300">
              <Text className="text-gray-400 font-medium">Before Image</Text>
            </View>
          )}
          
          {afterUri ? (
            <View className="flex-1">
              <RNImage 
                source={{ uri: afterUri }} 
                style={{ width: '100%', height: 160, borderRadius: 12 }}
                resizeMode="cover"
              />
              <View className="absolute top-2 left-2 bg-black/60 rounded-lg px-2 py-1">
                <Text className="text-white text-xs font-semibold">After</Text>
              </View>
            </View>
          ) : (
            <View className="flex-1 h-40 rounded-xl bg-gray-100 items-center justify-center border-2 border-dashed border-gray-300">
              <Text className="text-gray-400 font-medium">After Image</Text>
            </View>
          )}
        </View>

        {!!detectedRes && (
          <View className="mt-4 bg-blue-50 rounded-lg p-3 border border-blue-200">
            <Text className="text-sm text-blue-800">
              ✓ Detected resolution: <Text className="font-bold">{detectedRes} m/pixel</Text>
            </Text>
          </View>
        )}
      </View>

      {/* Analyze Button */}
      <TouchableOpacity 
        className={`rounded-xl py-4 px-6 shadow-lg mb-5 ${loading ? 'bg-green-400' : 'bg-green-600 active:bg-green-700'}`}
        onPress={onUpload} 
        disabled={loading}
      >
        {loading ? (
          <View className="flex-row items-center justify-center gap-2">
            <ActivityIndicator color="#fff"/>
            <Text className="text-white font-bold text-lg">Analyzing...</Text>
          </View>
        ) : (
          <Text className="text-white font-bold text-center text-lg">🔍 Start Analysis</Text>
        )}
      </TouchableOpacity>

      {/* Segmentation Overlays */}
      {(!!beforeOverlayUrl || !!afterOverlayUrl) && (
        <View className="bg-white rounded-2xl shadow-lg p-5 mb-5">
          <Text className="text-xl font-bold text-gray-800 mb-4">
            🎨 Segmentation Overlays
          </Text>
          <Text className="text-sm text-gray-600 mb-4">Color-coded by change type</Text>
          
          <View className="flex-col">
            {!!beforeOverlayUrl && (
              <View className="flex-1 mb-4">
                <View className="bg-gradient-to-r from-purple-100 to-blue-100 rounded-lg p-2 mb-2">
                  <Text className="text-xs font-semibold text-gray-700 text-center">
                    Before (with segments)
                  </Text>
                </View>
                <RNImage 
                  source={{ uri: beforeOverlayUrl }} 
                  style={beforeOverlayStyle}
                  resizeMode="contain"
                />
                <TouchableOpacity 
                  className="mt-3 bg-purple-600 rounded-xl py-3 px-4 shadow-md active:bg-purple-700"
                  onPress={onDownloadBeforeOverlay}
                >
                  <Text className="text-white font-semibold text-center">⬇️ Download Before Overlay</Text>
                </TouchableOpacity>
              </View>
            )}
            {!!afterOverlayUrl && (
              <View className="flex-1">
                <View className="bg-gradient-to-r from-blue-100 to-green-100 rounded-lg p-2 mb-2">
                  <Text className="text-xs font-semibold text-gray-700 text-center">
                    After (color by change)
                  </Text>
                </View>
                <RNImage 
                  source={{ uri: afterOverlayUrl }} 
                  style={afterOverlayStyle}
                  resizeMode="contain"
                />
                <TouchableOpacity 
                  className="mt-3 bg-green-600 rounded-xl py-3 px-4 shadow-md active:bg-green-700"
                  onPress={onDownloadAfterOverlay}
                >
                  <Text className="text-white font-semibold text-center">⬇️ Download After Overlay</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        </View>
      )}

      {/* Results Section */}
      {!!result && (
        <View className="bg-white rounded-2xl shadow-lg p-5 mb-5">
          <Text className="text-xl font-bold text-gray-800 mb-4">
            📊 Analysis Results
          </Text>

          {/* Object Count Cards */}
          <View className="flex-row gap-3 mb-4">
            <View className="flex-1 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 border border-blue-200">
              <Text className="text-xs text-blue-600 font-semibold mb-1">BEFORE</Text>
              <Text className="text-2xl font-bold text-blue-900">
                {metrics.total_objects_before ?? '—'}
              </Text>
              <Text className="text-xs text-blue-700 mt-1">Objects</Text>
            </View>
            <View className="flex-1 bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-xl p-4 border border-indigo-200">
              <Text className="text-xs text-indigo-600 font-semibold mb-1">AFTER</Text>
              <Text className="text-2xl font-bold text-indigo-900">
                {metrics.total_objects_after ?? '—'}
              </Text>
              <Text className="text-xs text-indigo-700 mt-1">Objects</Text>
            </View>
          </View>

          {/* Color-Coded Change Categories */}
          <Text className="text-base font-bold text-gray-700 mb-3 mt-2">
            🎨 Change Categories
          </Text>
          
          <View className="space-y-3 mb-4">
            {/* Flooded Land Area */}
            <View className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl p-4 border-2 border-blue-300">
              <View className="flex-row items-center mb-2">
                <View className="w-4 h-4 rounded-full bg-blue-500 mr-2"></View>
                <Text className="text-sm font-bold text-blue-900">💧 Flooded Land Area</Text>
              </View>
              <Text className="text-xs text-blue-700 mb-1">
                {fmt((metrics.total_flooded_pixels ?? metrics.new_water_pixels ?? pixelAnalysis.water_candidate_pixels), 0)} pixels
              </Text>
              <Text className="text-xl font-bold text-blue-900 mt-1">
                {fmt(areas.flooded_m2, 2)} m²
              </Text>
              <Text className="text-xs text-blue-600 mt-1">
                {fmt(areas.flooded_ha, 4)} hectares
              </Text>
            </View>

            {/* Vegetation Loss */}
            <View className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-4 border-2 border-green-300">
              <View className="flex-row items-center mb-2">
                <View className="w-4 h-4 rounded-full bg-green-600 mr-2"></View>
                <Text className="text-sm font-bold text-green-900">🌿 Vegetation Loss</Text>
              </View>
              <Text className="text-xs text-green-700 mb-1">
                {fmt((metrics.vegetation_loss_pixels ?? pixelAnalysis.veg_loss_pixels), 0)} pixels
              </Text>
              <Text className="text-xl font-bold text-green-900 mt-1">
                {fmt(areas.veg_m2, 2)} m²
              </Text>
              <Text className="text-xs text-green-600 mt-1">
                {fmt(areas.veg_ha, 4)} hectares
              </Text>
            </View>

            {/* Built Structures Affected */}
            <View className="bg-gradient-to-r from-orange-50 to-red-50 rounded-xl p-4 border-2 border-orange-300">
              <View className="flex-row items-center mb-2">
                <View className="w-4 h-4 rounded-full bg-orange-500 mr-2"></View>
                <Text className="text-sm font-bold text-orange-900">🏘️ Built Structures Affected</Text>
              </View>
              <Text className="text-xs text-orange-700 mb-1">
                {fmt(metrics.built_structures_affected_pixels, 0)} pixels
              </Text>
              <Text className="text-xl font-bold text-orange-900 mt-1">
                {fmt(areas.built_m2, 2)} m²
              </Text>
              <Text className="text-xs text-orange-600 mt-1">
                {fmt(areas.built_ha, 4)} hectares
              </Text>
            </View>
          </View>

          {/* Resolution Info */}
          {metrics.pixel_resolution_m && (
            <View className="bg-gray-50 rounded-lg p-3 border border-gray-200 mb-4">
              <Text className="text-xs text-gray-600">
                📏 Pixel resolution: <Text className="font-semibold text-gray-800">
                  {metrics.pixel_resolution_m?.toFixed?.(6)} m/pixel
                </Text>
              </Text>
            </View>
          )}

          {/* Quantify Change - computed from pixels and resolution */}
          {!!quantify && (
            <View className="bg-gradient-to-b from-emerald-50 to-white rounded-xl p-4 mb-4 border border-emerald-200">
              <Text className="text-base font-bold text-emerald-900 mb-3">🧮 Quantify Change</Text>

              {/* Flooded land (backend-preferred) */}
              <View className="mb-3">
                <Text className="text-sm font-semibold text-gray-800">💧 Area of flooded land</Text>
                <View className="flex-row items-baseline gap-2 mt-1">
                  <Text className="text-lg font-bold text-gray-900">
                    {fmt(areas.flooded_m2, 2)}
                  </Text>
                  <Text className="text-xs text-gray-600">m²</Text>
                </View>
                <Text className="text-xs text-gray-700 mt-0.5">
                  {fmt(areas.flooded_ha, 4)} hectares
                </Text>
              </View>

              {/* Vegetation lost (backend-preferred) */}
              <View className="mb-3">
                <Text className="text-sm font-semibold text-gray-800">🌿 Area of vegetation lost</Text>
                <View className="flex-row items-baseline gap-2 mt-1">
                  <Text className="text-lg font-bold text-gray-900">
                    {fmt(areas.veg_m2, 2)}
                  </Text>
                  <Text className="text-xs text-gray-600">m²</Text>
                </View>
                <Text className="text-xs text-gray-700 mt-0.5">
                  {fmt(areas.veg_ha, 4)} hectares
                </Text>
              </View>

              {/* Built structures affected (backend-preferred) */}
              <View>
                <Text className="text-sm font-semibold text-gray-800">🏘️ Area of affected built structures</Text>
                <View className="flex-row items-baseline gap-2 mt-1">
                  <Text className="text-lg font-bold text-gray-900">
                    {fmt(areas.built_m2, 2)}
                  </Text>
                  <Text className="text-xs text-gray-600">m²</Text>
                </View>
                <Text className="text-xs text-gray-700 mt-0.5">
                  {fmt(areas.built_ha, 4)} hectares
                </Text>
              </View>
            </View>
          )}

          {/* Color Legend */}
          <View className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl p-4 mb-4 border border-gray-200">
            <Text className="text-sm font-bold text-gray-800 mb-3">🎨 Color Legend</Text>
            <View className="space-y-2">
              <View className="flex-row items-center">
                <View className="w-3 h-3 rounded-full bg-blue-500 mr-2"></View>
                <Text className="text-xs text-gray-700">Blue = Water / Flooded areas</Text>
              </View>
              <View className="flex-row items-center">
                <View className="w-3 h-3 rounded-full bg-green-600 mr-2"></View>
                <Text className="text-xs text-gray-700">Green = Vegetation (lost areas)</Text>
              </View>
              <View className="flex-row items-center">
                <View className="w-3 h-3 rounded-full bg-orange-500 mr-2"></View>
                <Text className="text-xs text-gray-700">Orange = Built structures affected</Text>
              </View>
              <View className="flex-row items-center">
                <View className="w-3 h-3 rounded-full bg-gray-400 mr-2"></View>
                <Text className="text-xs text-gray-700">Gray = Bare soil / Unchanged</Text>
              </View>
            </View>
          </View>

          {/* Download Buttons */}
          <View className="flex-row gap-3 mt-2">
            <TouchableOpacity 
              className="flex-1 bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl py-3.5 px-4 shadow-md active:opacity-80"
              onPress={onOpenGeoJSON}
            >
              <Text className="text-white font-semibold text-center text-sm">
                📍 GeoJSON
              </Text>
            </TouchableOpacity>
            <TouchableOpacity 
              className="flex-1 bg-gradient-to-r from-green-500 to-green-600 rounded-xl py-3.5 px-4 shadow-md active:opacity-80"
              onPress={onOpenCSV}
            >
              <Text className="text-white font-semibold text-center text-sm">
                📄 CSV Report
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </ScrollView>
  );
}


const overlayStyles = StyleSheet.create({
  overlayImage: {
    width: '100%',
    // ensure the image doesn't overflow when using aspectRatio/contain
    overflow: 'hidden',
  },
  overlayFallback: {
    height: 180,
  },
});
