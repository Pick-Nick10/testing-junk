#pragma once

#ifdef USE_ESP32

#include "preprocessor_settings.h"
#include "streaming_model.h"

#include "esphome/components/microphone/microphone_source.h"

#include "esphome/core/automation.h"
#include "esphome/core/component.h"
#include "esphome/core/defines.h"
#include "esphome/core/ring_buffer.h"

#ifdef USE_OTA_STATE_LISTENER
#include "esphome/components/ota/ota_backend.h"
#endif

#include <atomic>

#include <freertos/event_groups.h>

#include <lwip/sockets.h>

#include <frontend.h>
#include <frontend_util.h>

namespace esphome {
namespace micro_wake_word {

enum State {
  STARTING,
  DETECTING_WAKE_WORD,
  STOPPING,
  STOPPED,
};

class MicroWakeWord : public Component
#ifdef USE_OTA_STATE_LISTENER
    ,
                      public ota::OTAGlobalStateListener
#endif
{
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override;
  void dump_config() override;

#ifdef USE_OTA_STATE_LISTENER
  void on_ota_global_state(ota::OTAState state, float progress, uint8_t error, ota::OTAComponent *comp) override;
#endif

  void start();
  void stop();

  bool is_running() const { return this->state_ != State::STOPPED; }

  void set_features_step_size(uint8_t step_size) { this->features_step_size_ = step_size; }

  void set_microphone_source(microphone::MicrophoneSource *microphone_source) {
    this->microphone_source_ = microphone_source;
  }

  void set_stop_after_detection(bool stop_after_detection) { this->stop_after_detection_ = stop_after_detection; }

  Trigger<std::string> *get_wake_word_detected_trigger() { return &this->wake_word_detected_trigger_; }

  void add_wake_word_model(WakeWordModel *model);

#ifdef USE_MICRO_WAKE_WORD_VAD
  void add_vad_model(const uint8_t *model_start, uint8_t probability_cutoff, size_t sliding_window_size,
                     size_t tensor_arena_size);

  // Intended for the voice assistant component to fetch VAD status
  bool get_vad_state() { return this->vad_state_; }
#endif

  // Intended for the voice assistant component to access which wake words are available
  // Since these are pointers to the WakeWordModel objects, the voice assistant component can enable or disable them
  std::vector<WakeWordModel *> get_wake_words();

  // Clip capture configuration
  void set_clip_receiver(const std::string &host, uint16_t port);
  void set_near_miss_threshold_factor(float factor);
  void set_clip_preroll_ms(uint32_t ms);
  void set_clip_postroll_ms(uint32_t ms);
  void set_clip_gain_factor(uint8_t gain) { this->clip_gain_factor_ = gain; }

 protected:
  microphone::MicrophoneSource *microphone_source_{nullptr};
  Trigger<std::string> wake_word_detected_trigger_;
  State state_{State::STOPPED};

  std::weak_ptr<RingBuffer> ring_buffer_;
  std::vector<WakeWordModel *> wake_word_models_;

#ifdef USE_MICRO_WAKE_WORD_VAD
  std::unique_ptr<VADModel> vad_model_;
  bool vad_state_{false};
#endif

  bool pending_start_{false};
  bool pending_stop_{false};

  bool stop_after_detection_;

  uint8_t features_step_size_;

  // Audio frontend handles generating spectrogram features
  struct FrontendConfig frontend_config_;
  struct FrontendState frontend_state_;

  // Handles managing the stop/state of the inference task
  EventGroupHandle_t event_group_;

  // Used to send messages about the models' states to the main loop
  QueueHandle_t detection_queue_;

  static void inference_task(void *params);
  TaskHandle_t inference_task_handle_{nullptr};

  /// @brief Suspends the inference task
  void suspend_task_();
  /// @brief Resumes the inference task
  void resume_task_();

  void set_state_(State state);

  /// @brief Generates spectrogram features from an input buffer of audio samples
  /// @param audio_buffer (int16_t *) Buffer containing input audio samples
  /// @param samples_available (size_t) Number of samples avaiable in the input buffer
  /// @param features_buffer (int8_t *) Buffer to store generated features
  /// @return (size_t) Number of samples processed from the input buffer
  size_t generate_features_(int16_t *audio_buffer, size_t samples_available,
                            int8_t features_buffer[PREPROCESSOR_FEATURE_SIZE]);

  /// @brief Processes any new probabilities for each model. If any wake word is detected, it will send a DetectionEvent
  /// to the detection_queue_.
  void process_probabilities_();

  /// @brief Deletes each model's TFLite interpreters and frees tensor arena memory.
  void unload_models_();

  /// @brief Runs an inference with each model using the new spectrogram features
  /// @param audio_features (int8_t *) Buffer containing new spectrogram features
  /// @return True if successful, false if any errors were encountered
  bool update_model_probabilities_(const int8_t audio_features[PREPROCESSOR_FEATURE_SIZE]);

  // --- Clip capture ---

  /// @brief Writes audio data to the circular clip buffer
  void write_to_clip_buffer_(const uint8_t *data, size_t len);
  /// @brief Initiates a clip capture with post-roll delay
  void trigger_clip_save_(const DetectionEvent &event);
  /// @brief Sends the clip buffer over UDP (called from send task)
  void send_clip_udp_();
  /// @brief Initializes the UDP socket
  bool init_clip_socket_();
  /// @brief FreeRTOS task that sends clips over UDP
  static void clip_send_task(void *params);

  // Circular buffer (PSRAM)
  uint8_t *clip_buffer_{nullptr};
  size_t clip_buffer_size_{0};
  size_t clip_write_pos_{0};

  // Linear send buffer (PSRAM) — filled by mic callback, consumed by send task
  uint8_t *clip_send_buffer_{nullptr};
  size_t clip_send_buffer_size_{0};

  // Clip state
  std::atomic<bool> clip_capture_pending_{false};
  std::atomic<uint32_t> clip_postroll_end_ms_{0};
  DetectionEvent pending_clip_event_;       // Written by inference task, read by mic callback
  DetectionEvent sending_clip_event_;       // Copied by mic callback, read by send task (safe: sequential)
  uint32_t clip_cooldown_until_ms_{0};

  // UDP
  int clip_udp_socket_{-1};
  struct sockaddr_in clip_udp_dest_;

  // Send task
  TaskHandle_t clip_send_task_handle_{nullptr};

  // Configuration
  std::string clip_receiver_host_;
  uint16_t clip_receiver_port_{0};
  float near_miss_threshold_factor_{0.5f};
  uint32_t clip_preroll_ms_{3000};
  uint32_t clip_postroll_ms_{500};
  uint8_t clip_gain_factor_{1};  // Gain applied to clip audio (1 = no gain)
};

}  // namespace micro_wake_word
}  // namespace esphome

#endif  // USE_ESP32
