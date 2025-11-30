import React from 'react';
import { View, Text, ScrollView } from 'react-native';

export default function ExploreScreen() {
  return (
    <ScrollView contentContainerStyle={{ padding: 20 }} className="flex-1 bg-white">
      <View className="mb-6">
        <Text className="text-3xl font-bold text-gray-900 mb-2">🔎 Explore</Text>
        <Text className="text-base text-gray-600">This is a placeholder Explore screen. Add map or data views here.</Text>
      </View>
      <View className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 rounded-2xl border border-blue-200">
        <Text className="text-sm text-gray-700 leading-5">
          You can now remove this file if you don't want an Explore tab, or replace this content with your desired UI (e.g., interactive map, historical flood events, imagery catalog, etc.).
        </Text>
      </View>
    </ScrollView>
  );
}
