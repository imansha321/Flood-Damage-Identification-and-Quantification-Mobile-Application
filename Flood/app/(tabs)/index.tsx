import React, { useCallback, useMemo, useState } from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, Alert, ScrollView, Linking } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Image } from 'expo-image';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

// Adjust this to your backend server URL
const API_BASE = 'http://localhost:8000';

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

  const pickImage = useCallback(async (setter: (uri: string) => void) => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 1,
    });
    if (!res.canceled && res.assets?.[0]?.uri) {
      setter(res.assets[0].uri);
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
      const beforeName = beforeUri.split('/').pop() || 'before.png';
      const afterName = afterUri.split('/').pop() || 'after.png';
      
      // Fetch the image URIs as blobs for web compatibility
      const beforeBlob = await fetch(beforeUri).then(r => r.blob());
      const afterBlob = await fetch(afterUri).then(r => r.blob());
      
      // Append as File objects (web) or fallback to RN format
      form.append('before', beforeBlob, beforeName);
      form.append('after', afterBlob, afterName);
      
      const url = `${API_BASE}/analyze/`;
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Accept': 'application/json' },
        body: form,
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `Upload failed: ${resp.status}`);
      }
      const data = await resp.json();
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

  const metrics = result?.metrics ?? {};
  
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

    const floodedPixels = toNum(metrics?.total_flooded_pixels ?? metrics?.new_water_pixels);
    const vegetationPixels = toNum(metrics?.vegetation_loss_pixels);
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
              <Image source={{ uri: beforeUri }} className="w-full h-40 rounded-xl" contentFit="cover"/>
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
              <Image source={{ uri: afterUri }} className="w-full h-40 rounded-xl" contentFit="cover"/>
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
          
          <View className="flex-row gap-3">
            {!!beforeOverlayUrl && (
              <View className="flex-1">
                <View className="bg-gradient-to-r from-purple-100 to-blue-100 rounded-lg p-2 mb-2">
                  <Text className="text-xs font-semibold text-gray-700 text-center">
                    Before (with segments)
                  </Text>
                </View>
                <Image 
                  source={{ uri: beforeOverlayUrl }} 
                  className="w-full h-48 rounded-xl border-2 border-gray-200" 
                  contentFit="contain"
                />
              </View>
            )}
            {!!afterOverlayUrl && (
              <View className="flex-1">
                <View className="bg-gradient-to-r from-blue-100 to-green-100 rounded-lg p-2 mb-2">
                  <Text className="text-xs font-semibold text-gray-700 text-center">
                    After (color by change)
                  </Text>
                </View>
                <Image 
                  source={{ uri: afterOverlayUrl }} 
                  className="w-full h-48 rounded-xl border-2 border-gray-200" 
                  contentFit="contain"
                />
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
                {metrics.total_flooded_pixels ?? metrics.new_water_pixels ?? '—'} pixels
              </Text>
              {metrics.pixel_resolution_m && (
                <>
                  <Text className="text-xl font-bold text-blue-900 mt-1">
                    {(metrics.total_flooded_area_m2 ?? metrics.new_water_area_m2)?.toFixed?.(2) ?? '—'} m²
                  </Text>
                  <Text className="text-xs text-blue-600 mt-1">
                    {(metrics.total_flooded_area_ha ?? metrics.new_water_area_ha)?.toFixed?.(4) ?? '—'} hectares
                  </Text>
                </>
              )}
            </View>

            {/* Vegetation Loss */}
            <View className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-4 border-2 border-green-300">
              <View className="flex-row items-center mb-2">
                <View className="w-4 h-4 rounded-full bg-green-600 mr-2"></View>
                <Text className="text-sm font-bold text-green-900">🌿 Vegetation Loss</Text>
              </View>
              <Text className="text-xs text-green-700 mb-1">
                {metrics.vegetation_loss_pixels ?? '—'} pixels
              </Text>
              {metrics.pixel_resolution_m && (
                <>
                  <Text className="text-xl font-bold text-green-900 mt-1">
                    {metrics.vegetation_loss_area_m2?.toFixed?.(2) ?? '—'} m²
                  </Text>
                  <Text className="text-xs text-green-600 mt-1">
                    {metrics.vegetation_loss_area_ha?.toFixed?.(4) ?? '—'} hectares
                  </Text>
                </>
              )}
            </View>

            {/* Built Structures Affected */}
            <View className="bg-gradient-to-r from-orange-50 to-red-50 rounded-xl p-4 border-2 border-orange-300">
              <View className="flex-row items-center mb-2">
                <View className="w-4 h-4 rounded-full bg-orange-500 mr-2"></View>
                <Text className="text-sm font-bold text-orange-900">🏘️ Built Structures Affected</Text>
              </View>
              <Text className="text-xs text-orange-700 mb-1">
                {metrics.built_structures_affected_pixels ?? '—'} pixels
              </Text>
              {metrics.pixel_resolution_m && (
                <>
                  <Text className="text-xl font-bold text-orange-900 mt-1">
                    {metrics.built_structures_affected_area_m2?.toFixed?.(2) ?? '—'} m²
                  </Text>
                  <Text className="text-xs text-orange-600 mt-1">
                    {metrics.built_structures_affected_area_ha?.toFixed?.(4) ?? '—'} hectares
                  </Text>
                </>
              )}
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

              {/* Flooded land */}
              <View className="mb-3">
                <Text className="text-sm font-semibold text-gray-800">💧 Area of flooded land</Text>
                <View className="flex-row items-baseline gap-2 mt-1">
                  <Text className="text-lg font-bold text-gray-900">
                    {quantify.flooded?.m2?.toFixed?.(2) ?? '—'}
                  </Text>
                  <Text className="text-xs text-gray-600">m²</Text>
                </View>
                <Text className="text-xs text-gray-700 mt-0.5">
                  {quantify.flooded?.ha?.toFixed?.(4) ?? '—'} hectares
                </Text>
              </View>

              {/* Vegetation lost */}
              <View className="mb-3">
                <Text className="text-sm font-semibold text-gray-800">🌿 Area of vegetation lost</Text>
                <View className="flex-row items-baseline gap-2 mt-1">
                  <Text className="text-lg font-bold text-gray-900">
                    {quantify.vegetation?.m2?.toFixed?.(2) ?? '—'}
                  </Text>
                  <Text className="text-xs text-gray-600">m²</Text>
                </View>
                <Text className="text-xs text-gray-700 mt-0.5">
                  {quantify.vegetation?.ha?.toFixed?.(4) ?? '—'} hectares
                </Text>
              </View>

              {/* Built structures affected */}
              <View>
                <Text className="text-sm font-semibold text-gray-800">🏘️ Area of affected built structures</Text>
                <View className="flex-row items-baseline gap-2 mt-1">
                  <Text className="text-lg font-bold text-gray-900">
                    {quantify.built?.m2?.toFixed?.(2) ?? '—'}
                  </Text>
                  <Text className="text-xs text-gray-600">m²</Text>
                </View>
                <Text className="text-xs text-gray-700 mt-0.5">
                  {quantify.built?.ha?.toFixed?.(4) ?? '—'} hectares
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
