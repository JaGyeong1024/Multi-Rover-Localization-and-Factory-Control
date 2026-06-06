// ROS connection layer (roslib over rosbridge).
//
// Provides:
//   - RosProvider     : owns a single ROSLIB.Ros connection with auto-reconnect.
//   - useRos()        : { ros, connected, url } — connection handle + state.
//   - useSubscription(topic, type, cb, deps) : subscribe helper with cleanup.
//   - usePublisher(topic, type) : returns a publish(messageObject) function.
//
// The single Ros instance lives for the app lifetime; on 'close' we retry every
// 2s. Subscriptions are created against the *current* ros object and torn down
// on unmount or when ros is replaced (reconnect creates a fresh object).

import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
} from 'react';
import ROSLIB from 'roslib';

const ROSBRIDGE_URL = 'ws://localhost:9090';
const RECONNECT_MS = 2000;

const RosContext = createContext({ ros: null, connected: false, url: ROSBRIDGE_URL });

export function RosProvider({ children, url = ROSBRIDGE_URL }) {
  const [connected, setConnected] = useState(false);
  // `ros` is held in state so consumers re-render (and re-subscribe) when a
  // reconnect swaps in a fresh ROSLIB.Ros object.
  const [ros, setRos] = useState(null);
  const retryTimer = useRef(null);
  const closedByUnmount = useRef(false);

  useEffect(() => {
    closedByUnmount.current = false;

    function connect() {
      const r = new ROSLIB.Ros({ url });

      r.on('connection', () => {
        setConnected(true);
      });

      r.on('error', () => {
        // roslib also emits 'close' after an error; reconnect is handled there.
        setConnected(false);
      });

      r.on('close', () => {
        setConnected(false);
        if (closedByUnmount.current) return;
        // Schedule a reconnect with a brand-new Ros object.
        if (retryTimer.current) clearTimeout(retryTimer.current);
        retryTimer.current = setTimeout(connect, RECONNECT_MS);
      });

      setRos(r);
    }

    connect();

    return () => {
      closedByUnmount.current = true;
      if (retryTimer.current) clearTimeout(retryTimer.current);
      setRos((prev) => {
        try {
          prev && prev.close();
        } catch (e) {
          /* ignore */
        }
        return null;
      });
    };
  }, [url]);

  return (
    <RosContext.Provider value={{ ros, connected, url }}>
      {children}
    </RosContext.Provider>
  );
}

export function useRos() {
  return useContext(RosContext);
}

/**
 * Subscribe to a topic. The callback fires with the raw ROS message.
 * Re-subscribes whenever `ros` changes (reconnect) or `deps` change.
 */
export function useSubscription(topic, messageType, callback, deps = []) {
  const { ros } = useRos();
  const cbRef = useRef(callback);
  cbRef.current = callback;

  useEffect(() => {
    if (!ros) return undefined;

    const sub = new ROSLIB.Topic({ ros, name: topic, messageType });
    const handler = (msg) => cbRef.current(msg);
    sub.subscribe(handler);

    return () => {
      try {
        sub.unsubscribe(handler);
      } catch (e) {
        /* ignore */
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ros, topic, messageType, ...deps]);
}

/**
 * Returns a stable publish(messageObject) function for an advertised topic.
 * The Topic is (re)created when `ros` changes.
 */
export function usePublisher(topic, messageType) {
  const { ros } = useRos();
  const topicRef = useRef(null);

  useEffect(() => {
    if (!ros) {
      topicRef.current = null;
      return undefined;
    }
    const t = new ROSLIB.Topic({ ros, name: topic, messageType });
    t.advertise();
    topicRef.current = t;
    return () => {
      try {
        t.unadvertise();
      } catch (e) {
        /* ignore */
      }
      topicRef.current = null;
    };
  }, [ros, topic, messageType]);

  return useCallback((messageObject) => {
    if (!topicRef.current) return false;
    topicRef.current.publish(new ROSLIB.Message(messageObject));
    return true;
  }, []);
}
