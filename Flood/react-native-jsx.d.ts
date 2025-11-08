// Temporary override to unblock JSX compilation issues stemming from React/React Native version/type mismatches.
// NOTE: This should be removed once dependencies are aligned (React 18.2 + react-native 0.76 + matching @types/react).
// Provides permissive IntrinsicElements definitions.

declare namespace JSX {
  interface IntrinsicElements {
    ScrollView: any;
    View: any;
    Text: any;
    Image: any;
    TouchableOpacity: any;
    ActivityIndicator: any;
  }
}
