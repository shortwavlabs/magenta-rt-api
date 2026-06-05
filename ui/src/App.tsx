import {
  Activity,
  AudioLines,
  CheckCircle2,
  Cloud,
  Download,
  FileArchive,
  FileAudio2,
  Loader2,
  Music2,
  Play,
  Radio,
  RefreshCw,
  SlidersHorizontal,
  Upload,
  Wand2,
} from "lucide-react";
import {
  type ChangeEvent,
  type CSSProperties,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { joinUrl, parseFilename, readApiError } from "@/lib/utils";

type Mode = "text" | "variation" | "inpaint";
type JobStatus = "queued" | "running" | "succeeded" | "failed";
type WaveformStatus = "standby" | "rendering" | "decoding" | "decoded" | "unsupported" | "failed";

interface WaveformBar {
  peak: number;
  rms: number;
}

interface WaveformAnalysis {
  bars: WaveformBar[];
  duration: number;
  sampleRate: number;
  channels: number;
}

interface ModelAssetsResponse {
  magenta_home: string | null;
  resources_ready: boolean;
  models: Record<string, boolean>;
  details: Record<string, string>;
}

interface HealthResponse {
  status: string;
  model: string;
  backend: string;
  loaded: boolean;
  storage_backend: string;
  available_models: string[];
  available_backends: string[];
  loaded_models: string[];
  preload_models: string[];
  model_duration_limits_seconds: Record<string, number>;
  max_duration_seconds: number;
  max_batch_size: number;
  sample_rate: number;
  frames_per_second: number;
  assets: ModelAssetsResponse;
  load_error: string | null;
}

interface JobResponse {
  id: string;
  status: JobStatus;
  mode: string;
  model: string;
  backend: string;
  duration: number;
  frames: number;
  output_count: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  download_url: string | null;
  download_content_type: string | null;
  error: string | null;
  storage_backend: string | null;
  storage_key: string | null;
  sample_rate: number | null;
}

interface LocalArtifact {
  url: string;
  filename: string;
  contentType: string;
  outputCount: number;
  createdAt: string;
}

interface FormState {
  model: string;
  backend: string;
  prompt: string;
  duration: number;
  temperature: number;
  topK: number;
  cfgMusiccoca: number;
  cfgNotes: number;
  cfgDrums: number;
  batchSize: number;
  seed: number;
  audioStyleWeight: number;
}

const API_BASE_STORAGE_KEY = "magenta-rt-ui-api-base";
const DEFAULT_API_BASE = import.meta.env.VITE_MAGENTA_RT_API_BASE_URL ?? "/api";

const waveformLevels = [
  0.2, 0.38, 0.7, 0.48, 0.82, 0.36, 0.24, 0.62, 0.9, 0.54,
  0.33, 0.74, 0.45, 0.3, 0.65, 0.86, 0.52, 0.4, 0.72, 0.95,
  0.78, 0.44, 0.32, 0.68, 0.88, 0.56, 0.28, 0.6, 0.76, 0.42,
  0.25, 0.66, 0.84, 0.5, 0.36, 0.58, 0.73, 0.46, 0.3, 0.52,
];
const placeholderWaveformBars = waveformLevels.map((level) => ({
  peak: level,
  rms: Math.max(0.08, level * 0.68),
}));

const initialForm: FormState = {
  model: "mrt2_small",
  backend: "mlxfn",
  prompt: "disco funk bassline, crisp drums, warm electric piano",
  duration: 4,
  temperature: 1.3,
  topK: 40,
  cfgMusiccoca: 3,
  cfgNotes: 1,
  cfgDrums: 1,
  batchSize: 1,
  seed: 0,
  audioStyleWeight: 0.55,
};

function App() {
  const [apiBase, setApiBase] = useState(() => {
    return window.localStorage.getItem(API_BASE_STORAGE_KEY) ?? DEFAULT_API_BASE;
  });
  const [mode, setMode] = useState<Mode>("text");
  const [form, setForm] = useState<FormState>(initialForm);
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [artifact, setArtifact] = useState<LocalArtifact | null>(null);
  const [waveformAnalysis, setWaveformAnalysis] = useState<WaveformAnalysis | null>(null);
  const [waveformStatus, setWaveformStatus] = useState<WaveformStatus>("standby");
  const [waveformError, setWaveformError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<"sync" | "job" | "health" | null>(null);
  const artifactUrlRef = useRef<string | null>(null);

  const availableModels = health?.available_models.length
    ? health.available_models
    : ["mrt2_small", "mrt2_base"];
  const availableBackends = health?.available_backends.length
    ? health.available_backends
    : ["mlxfn", "mlx", "jax"];
  const maxDuration = health?.model_duration_limits_seconds[form.model] ?? 60;
  const canUseSourceMode = mode === "text" || sourceFile !== null;
  const busy = busyAction !== null;
  const activeJobs = useMemo(
    () => jobs.filter((job) => job.status === "queued" || job.status === "running"),
    [jobs],
  );
  const waveformBars = waveformAnalysis?.bars ?? placeholderWaveformBars;

  const updateForm = <Key extends keyof FormState>(key: Key, value: FormState[Key]) => {
    setForm((current) => ({ ...current, [key]: value }));
  };

  const fetchHealth = useCallback(async () => {
    setBusyAction((current) => current ?? "health");
    try {
      const response = await fetch(joinUrl(apiBase, "/health"));
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }
      const nextHealth = (await response.json()) as HealthResponse;
      setHealth(nextHealth);
      setError(nextHealth.load_error);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyAction((current) => (current === "health" ? null : current));
    }
  }, [apiBase]);

  useEffect(() => {
    window.localStorage.setItem(API_BASE_STORAGE_KEY, apiBase);
  }, [apiBase]);

  useEffect(() => {
    void fetchHealth();
  }, [fetchHealth]);

  useEffect(() => {
    return () => {
      if (artifactUrlRef.current) {
        URL.revokeObjectURL(artifactUrlRef.current);
      }
    };
  }, []);

  const setLocalArtifact = (nextArtifact: LocalArtifact) => {
    if (artifactUrlRef.current) {
      URL.revokeObjectURL(artifactUrlRef.current);
    }
    artifactUrlRef.current = nextArtifact.url;
    setArtifact(nextArtifact);
  };

  const commonJsonPayload = () => {
    return {
      model: form.model,
      backend: form.backend,
      prompt: form.prompt,
      duration: form.duration,
      temperature: form.temperature,
      top_k: form.topK,
      cfg_musiccoca: form.cfgMusiccoca,
      cfg_notes: form.cfgNotes,
      cfg_drums: form.cfgDrums,
      batch_size: form.batchSize,
      seed: form.seed,
      audio_style_weight: form.audioStyleWeight,
    };
  };

  const commonFormData = () => {
    if (!sourceFile) {
      throw new Error("Source audio is required");
    }

    const data = new FormData();
    data.set("audio", sourceFile);
    data.set("model", form.model);
    data.set("backend", form.backend);
    data.set("prompt", form.prompt);
    data.set("duration", String(form.duration));
    data.set("temperature", String(form.temperature));
    data.set("top_k", String(form.topK));
    data.set("cfg_musiccoca", String(form.cfgMusiccoca));
    data.set("cfg_notes", String(form.cfgNotes));
    data.set("cfg_drums", String(form.cfgDrums));
    data.set("batch_size", String(form.batchSize));
    data.set("seed", String(form.seed));
    data.set("audio_style_weight", String(form.audioStyleWeight));
    return data;
  };

  const requestForMode = (asJob: boolean) => {
    if (mode === "text") {
      return {
        url: joinUrl(apiBase, asJob ? "/jobs" : "/v1/audio/generations"),
        init: {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(commonJsonPayload()),
        },
      };
    }

    const data = commonFormData();
    if (mode === "variation") {
      return {
        url: joinUrl(apiBase, asJob ? "/jobs/variations" : "/v1/audio/variations"),
        init: { method: "POST", body: data },
      };
    }

    return {
      url: joinUrl(apiBase, asJob ? "/jobs/inpaint" : "/v1/audio/inpaint"),
      init: { method: "POST", body: data },
    };
  };

  const runSync = async () => {
    setBusyAction("sync");
    setError(null);
    setWaveformAnalysis(null);
    setWaveformStatus("rendering");
    setWaveformError(null);
    try {
      const request = requestForMode(false);
      const response = await fetch(request.url, request.init);
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }

      const blob = await response.blob();
      const contentType = response.headers.get("content-type") ?? blob.type;
      const outputCount = Number(response.headers.get("x-output-count") ?? "1");
      const fallback = contentType.includes("zip") ? "magenta-rt-batch.zip" : "magenta-rt.wav";
      const filename = parseFilename(response.headers.get("content-disposition"), fallback);
      const url = URL.createObjectURL(blob);
      setLocalArtifact({
        url,
        filename,
        contentType,
        outputCount,
        createdAt: new Date().toISOString(),
      });

      if (isAudioArtifact(contentType, filename)) {
        setWaveformStatus("decoding");
        const analysis = await analyzeAudioBlob(blob);
        setWaveformAnalysis(analysis);
        setWaveformStatus("decoded");
      } else {
        setWaveformAnalysis(null);
        setWaveformStatus("unsupported");
      }
    } catch (caught) {
      setWaveformStatus("failed");
      setWaveformError(caught instanceof Error ? caught.message : String(caught));
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyAction(null);
    }
  };

  const pollJob = useCallback(
    async (jobId: string) => {
      try {
        const response = await fetch(joinUrl(apiBase, `/jobs/${jobId}`));
        if (!response.ok) {
          throw new Error(await readApiError(response));
        }
        const nextJob = (await response.json()) as JobResponse;
        setJobs((current) => {
          const exists = current.some((job) => job.id === nextJob.id);
          if (!exists) {
            return [nextJob, ...current];
          }
          return current.map((job) => (job.id === nextJob.id ? nextJob : job));
        });
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    },
    [apiBase],
  );

  const startJob = async () => {
    setBusyAction("job");
    setError(null);
    try {
      const request = requestForMode(true);
      const response = await fetch(request.url, request.init);
      if (!response.ok) {
        throw new Error(await readApiError(response));
      }
      const created = (await response.json()) as { id: string; status: JobStatus };
      const pendingJob: JobResponse = {
        id: created.id,
        status: created.status,
        mode: modeLabel(mode),
        model: form.model,
        backend: form.backend,
        duration: form.duration,
        frames: Math.max(1, Math.round(form.duration * (health?.frames_per_second ?? 25))),
        output_count: null,
        created_at: new Date().toISOString(),
        started_at: null,
        completed_at: null,
        download_url: null,
        download_content_type: null,
        error: null,
        storage_backend: null,
        storage_key: null,
        sample_rate: null,
      };
      setJobs((current) => [pendingJob, ...current.filter((job) => job.id !== created.id)]);
      await pollJob(created.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusyAction(null);
    }
  };

  const refreshJobs = useCallback(async () => {
    await Promise.all(jobs.map((job) => pollJob(job.id)));
  }, [jobs, pollJob]);

  useEffect(() => {
    if (activeJobs.length === 0) {
      return;
    }
    const interval = window.setInterval(() => {
      for (const job of activeJobs) {
        void pollJob(job.id);
      }
    }, 3500);
    return () => window.clearInterval(interval);
  }, [activeJobs, pollJob]);

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSourceFile(event.target.files?.[0] ?? null);
  };

  return (
    <main className="min-h-screen px-4 py-5 md:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <header className="rounded-lg border bg-card/85 p-4 shadow-sm backdrop-blur">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-3">
              <div className="flex size-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Music2 className="size-5" />
              </div>
              <div>
                <h1 className="text-xl font-semibold tracking-normal">Magenta RT 2 Testbench</h1>
                <div className="mt-1 flex flex-wrap items-center gap-2">
                  <StatusBadge status={health?.status ?? "offline"} />
                  <Badge variant="outline">{health?.backend ?? form.backend}</Badge>
                  <Badge variant="outline">{health?.storage_backend ?? "storage"}</Badge>
                  <Badge variant={health?.assets.resources_ready ? "success" : "warning"}>
                    resources
                  </Badge>
                </div>
              </div>
            </div>
            <div className="flex w-full flex-col gap-2 sm:flex-row lg:max-w-xl">
              <Input
                aria-label="API base URL"
                value={apiBase}
                onChange={(event) => setApiBase(event.target.value)}
              />
              <Button
                type="button"
                variant="outline"
                onClick={() => void fetchHealth()}
                disabled={busyAction === "health"}
              >
                {busyAction === "health" ? (
                  <Loader2 className="animate-spin" />
                ) : (
                  <RefreshCw />
                )}
                Health
              </Button>
            </div>
          </div>
        </header>

        {error ? (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            {error}
          </div>
        ) : null}

        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_390px]">
          <Card>
            <CardHeader className="pb-4">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div>
                  <CardTitle>Request</CardTitle>
                  <CardDescription>{modeLabel(mode)}</CardDescription>
                </div>
                <Badge variant="secondary">
                  {form.model} / {form.backend}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <Tabs value={mode} onValueChange={(value) => setMode(value as Mode)}>
                <TabsList className="!grid w-full grid-cols-3">
                  <TabsTrigger value="text">
                    <Wand2 className="mr-1.5" />
                    <span className="block min-w-0 truncate">Text</span>
                  </TabsTrigger>
                  <TabsTrigger value="variation">
                    <Upload className="mr-1.5" />
                    <span className="block min-w-0 truncate">Style</span>
                  </TabsTrigger>
                  <TabsTrigger value="inpaint">
                    <SlidersHorizontal className="mr-1.5" />
                    <span className="block min-w-0 truncate">Inpaint</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value={mode} className="space-y-5">
                  <div className="grid gap-4 lg:grid-cols-[1fr_180px_160px]">
                    <Field label="Prompt">
                      <Textarea
                        value={form.prompt}
                        onChange={(event) => updateForm("prompt", event.target.value)}
                      />
                    </Field>
                    <Field label="Model">
                      <select
                        className="h-9 w-full rounded-md border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        value={form.model}
                        onChange={(event) => updateForm("model", event.target.value)}
                      >
                        {availableModels.map((model) => (
                          <option key={model} value={model}>
                            {model}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <Field label="Backend">
                      <select
                        className="h-9 w-full rounded-md border bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        value={form.backend}
                        onChange={(event) => updateForm("backend", event.target.value)}
                      >
                        {availableBackends.map((backend) => (
                          <option key={backend} value={backend}>
                            {backend}
                          </option>
                        ))}
                      </select>
                    </Field>
                  </div>

                  {mode !== "text" ? (
                    <Field label="Source audio">
                      <div className="flex flex-col gap-2 sm:flex-row">
                        <Input
                          type="file"
                          accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a"
                          onChange={onFileChange}
                        />
                        {sourceFile ? (
                          <Badge variant="outline" className="h-9 justify-center">
                            {sourceFile.name}
                          </Badge>
                        ) : null}
                      </div>
                    </Field>
                  ) : null}

                  {mode !== "text" ? (
                    <Field label={`Audio style weight: ${form.audioStyleWeight.toFixed(2)}`}>
                      <Input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={form.audioStyleWeight}
                        onChange={(event) => {
                          updateForm("audioStyleWeight", Number(event.target.value));
                        }}
                      />
                    </Field>
                  ) : null}

                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                    <NumberField
                      label="Duration"
                      value={form.duration}
                      min={0.2}
                      max={maxDuration}
                      step={0.2}
                      onChange={(value) => updateForm("duration", value)}
                    />
                    <NumberField
                      label="Temperature"
                      value={form.temperature}
                      min={0.1}
                      max={5}
                      step={0.05}
                      onChange={(value) => updateForm("temperature", value)}
                    />
                    <NumberField
                      label="Top K"
                      value={form.topK}
                      min={0}
                      max={1024}
                      step={1}
                      onChange={(value) => updateForm("topK", value)}
                    />
                    <NumberField
                      label="Seed"
                      value={form.seed}
                      min={0}
                      step={1}
                      onChange={(value) => updateForm("seed", value)}
                    />
                  </div>

                  <details className="rounded-lg border bg-muted/30 p-4">
                    <summary className="flex cursor-pointer items-center gap-2 text-sm font-medium">
                      <SlidersHorizontal className="size-4" />
                      Advanced
                    </summary>
                    <div className="mt-4 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                      <NumberField
                        label="CFG MusicCoCa"
                        value={form.cfgMusiccoca}
                        min={-1}
                        max={7}
                        step={0.2}
                        onChange={(value) => updateForm("cfgMusiccoca", value)}
                      />
                      <NumberField
                        label="CFG Notes"
                        value={form.cfgNotes}
                        min={-1}
                        max={7}
                        step={0.2}
                        onChange={(value) => updateForm("cfgNotes", value)}
                      />
                      <NumberField
                        label="CFG Drums"
                        value={form.cfgDrums}
                        min={-1}
                        max={7}
                        step={0.2}
                        onChange={(value) => updateForm("cfgDrums", value)}
                      />
                      <NumberField
                        label="Batch"
                        value={form.batchSize}
                        min={1}
                        max={health?.max_batch_size ?? 4}
                        step={1}
                        onChange={(value) => updateForm("batchSize", value)}
                      />
                    </div>
                  </details>

                  <div className="flex flex-col gap-2 border-t pt-5 sm:flex-row">
                    <Button
                      type="button"
                      size="lg"
                      onClick={() => void runSync()}
                      disabled={busy || !canUseSourceMode}
                    >
                      {busyAction === "sync" ? <Loader2 className="animate-spin" /> : <Play />}
                      Generate
                    </Button>
                    <Button
                      type="button"
                      size="lg"
                      variant="secondary"
                      onClick={() => void startJob()}
                      disabled={busy || !canUseSourceMode}
                    >
                      {busyAction === "job" ? <Loader2 className="animate-spin" /> : <Cloud />}
                      Queue
                    </Button>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <aside className="space-y-5">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="size-4" />
                  Runtime
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Progress value={healthProgress(health)} />
                <Metric label="Default" value={health?.model ?? "unknown"} />
                <Metric label="Backend" value={health?.backend ?? form.backend} />
                <Metric label="Loaded" value={health?.loaded_models.join(", ") || "none"} />
                <Metric label="Preload" value={health?.preload_models.join(", ") || "none"} />
                <Separator />
                <div className="grid grid-cols-2 gap-2">
                  {["mrt2_small", "mrt2_base"].map((model) => (
                    <div key={model} className="rounded-lg border bg-background p-2 text-center">
                      <div className="text-[11px] font-medium text-muted-foreground">{model}</div>
                      <div className="mt-1 text-sm font-semibold">
                        {health?.assets.models[model] ? "ready" : "missing"}
                      </div>
                    </div>
                  ))}
                </div>
                <Metric
                  label="Resources"
                  value={health?.assets.resources_ready ? "ready" : "missing"}
                />
                <Metric
                  label="Sample rate"
                  value={health ? `${(health.sample_rate / 1000).toFixed(1)} kHz` : "-"}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center justify-between gap-3">
                  <CardTitle className="flex items-center gap-2">
                    <AudioLines className="size-4" />
                    Output
                  </CardTitle>
                  <Badge variant={artifact ? "success" : "outline"}>
                    {artifact ? "rendered" : "armed"}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="output-monitor">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-primary">
                        sample bus
                      </div>
                      <div className="mt-1 truncate text-sm font-semibold">
                        {artifact?.filename ?? modeLabel(mode)}
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <Radio className="size-3.5" />
                      {artifact ? formatArtifactTime(artifact.createdAt) : "standby"}
                    </div>
                  </div>

                  <div className="output-waveform">
                    <div className="output-playhead" />
                    {waveformBars.map((bar, index) => (
                      <div
                        className="output-wavebar"
                        key={index}
                        style={
                          {
                            "--peak": bar.peak,
                            "--rms": bar.rms,
                          } as CSSProperties
                        }
                      >
                        <span />
                      </div>
                    ))}
                  </div>

                  {artifact ? (
                    <div className="space-y-3">
                      <div className="grid grid-cols-3 gap-2">
                        <OutputMetric
                          label="format"
                          value={artifact.contentType.includes("zip") ? "ZIP" : "WAV"}
                        />
                        <OutputMetric label="clips" value={String(artifact.outputCount)} />
                        <OutputMetric label="source" value={mode} />
                      </div>
                      <WaveformReadout
                        analysis={waveformAnalysis}
                        error={waveformError}
                        status={waveformStatus}
                      />
                      {artifact.contentType.includes("audio") ? (
                        <audio className="w-full" src={artifact.url} controls />
                      ) : null}
                      <Button asChild className="w-full">
                        <a href={artifact.url} download={artifact.filename}>
                          {artifact.contentType.includes("zip") ? <FileArchive /> : <Download />}
                          {artifact.filename}
                        </a>
                      </Button>
                    </div>
                  ) : (
                    <div className="output-empty">
                      <FileAudio2 className="size-4" />
                      <span>{waveformStatusLabel(waveformStatus)}</span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center justify-between gap-3">
                  <CardTitle className="flex items-center gap-2">
                    <Cloud className="size-4" />
                    Jobs
                  </CardTitle>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => void refreshJobs()}
                    disabled={jobs.length === 0}
                    aria-label="Refresh jobs"
                  >
                    <RefreshCw />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {jobs.length === 0 ? (
                  <div className="rounded-lg border border-dashed p-5 text-center text-sm text-muted-foreground">
                    No jobs
                  </div>
                ) : (
                  jobs.map((job) => <JobRow key={job.id} job={job} />)
                )}
              </CardContent>
            </Card>
          </aside>
        </div>
      </div>
    </main>
  );
}

function Field({
  label,
  children,
  className,
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`space-y-2 ${className ?? ""}`}>
      <Label>{label}</Label>
      {children}
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <Field label={label}>
      <Input
        type="number"
        value={Number.isFinite(value) ? value : ""}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </Field>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="truncate font-medium">{value}</span>
    </div>
  );
}

function OutputMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border bg-background/70 px-2 py-2">
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 truncate text-sm font-semibold">{value}</div>
    </div>
  );
}

function WaveformReadout({
  analysis,
  error,
  status,
}: {
  analysis: WaveformAnalysis | null;
  error: string | null;
  status: WaveformStatus;
}) {
  if (analysis) {
    return (
      <div className="grid grid-cols-3 gap-2">
        <OutputMetric label="length" value={formatDuration(analysis.duration)} />
        <OutputMetric label="rate" value={`${(analysis.sampleRate / 1000).toFixed(1)} kHz`} />
        <OutputMetric label="ch" value={String(analysis.channels)} />
      </div>
    );
  }

  return (
    <div className="rounded-md border bg-background/70 px-3 py-2 text-xs text-muted-foreground">
      {status === "failed" && error ? error : waveformStatusLabel(status)}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  if (status === "ok") {
    return (
      <Badge variant="success">
        <CheckCircle2 className="mr-1 size-3" />
        ok
      </Badge>
    );
  }
  if (status === "loading") {
    return (
      <Badge variant="warning">
        <Loader2 className="mr-1 size-3 animate-spin" />
        loading
      </Badge>
    );
  }
  if (status === "error") {
    return <Badge variant="destructive">error</Badge>;
  }
  return <Badge variant="destructive">offline</Badge>;
}

function JobRow({ job }: { job: JobResponse }) {
  const contentType = job.download_content_type ?? "";
  const isAudio = contentType.includes("audio") && job.download_url;
  const isZip = contentType.includes("zip");

  return (
    <div className="rounded-lg border bg-background p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium">{job.id}</div>
          <div className="mt-1 text-xs text-muted-foreground">
            {job.mode} · {job.model} · {job.duration}s
          </div>
        </div>
        <JobBadge status={job.status} />
      </div>
      {job.error ? <p className="mt-2 text-xs text-red-700">{job.error}</p> : null}
      {isAudio ? <audio className="mt-3 w-full" src={job.download_url ?? ""} controls /> : null}
      {job.download_url ? (
        <Button asChild variant="outline" size="sm" className="mt-3 w-full">
          <a href={job.download_url} target="_blank" rel="noreferrer">
            {isZip ? <FileArchive /> : <Download />}
            Download
          </a>
        </Button>
      ) : null}
    </div>
  );
}

function JobBadge({ status }: { status: JobStatus }) {
  if (status === "succeeded") {
    return <Badge variant="success">done</Badge>;
  }
  if (status === "failed") {
    return <Badge variant="destructive">failed</Badge>;
  }
  if (status === "running") {
    return (
      <Badge variant="warning">
        <Loader2 className="mr-1 size-3 animate-spin" />
        running
      </Badge>
    );
  }
  return <Badge variant="outline">queued</Badge>;
}

function modeLabel(mode: Mode) {
  if (mode === "variation") {
    return "Audio style transfer";
  }
  if (mode === "inpaint") {
    return "Inpainting compatibility";
  }
  return "Text-to-audio generation";
}

function waveformStatusLabel(status: WaveformStatus) {
  if (status === "rendering") {
    return "Rendering audio";
  }
  if (status === "decoding") {
    return "Decoding waveform";
  }
  if (status === "decoded") {
    return "Decoded from returned WAV";
  }
  if (status === "unsupported") {
    return "Batch ZIP, download to inspect individual WAVs";
  }
  if (status === "failed") {
    return "Waveform decode failed";
  }
  return "Waiting for render";
}

function healthProgress(health: HealthResponse | null) {
  if (!health) {
    return 20;
  }
  if (health.status === "error") {
    return 12;
  }
  if (health.loaded) {
    return 100;
  }
  return health.assets.resources_ready ? 62 : 34;
}

function formatArtifactTime(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatDuration(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  if (minutes === 0) {
    return `${remainder.toFixed(1)}s`;
  }
  return `${minutes}:${String(Math.round(remainder)).padStart(2, "0")}`;
}

function isAudioArtifact(contentType: string, filename: string) {
  return contentType.includes("audio") || filename.toLowerCase().endsWith(".wav");
}

async function analyzeAudioBlob(blob: Blob, barCount = 40): Promise<WaveformAnalysis> {
  const AudioContextClass =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;

  if (!AudioContextClass) {
    throw new Error("This browser does not support AudioContext decoding.");
  }

  const audioContext = new AudioContextClass();
  try {
    const audioBuffer = await audioContext.decodeAudioData(await blob.arrayBuffer());
    return sampleAudioBuffer(audioBuffer, barCount);
  } finally {
    if (audioContext.state !== "closed") {
      await audioContext.close().catch(() => undefined);
    }
  }
}

function sampleAudioBuffer(audioBuffer: AudioBuffer, barCount: number): WaveformAnalysis {
  const rawBars: WaveformBar[] = [];
  let maxPeak = 0;

  for (let index = 0; index < barCount; index += 1) {
    const start = Math.floor((index * audioBuffer.length) / barCount);
    const end = Math.max(start + 1, Math.floor(((index + 1) * audioBuffer.length) / barCount));
    let peak = 0;
    let sumSquares = 0;
    let sampleCount = 0;

    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
      const channelData = audioBuffer.getChannelData(channel);
      for (let sample = start; sample < end; sample += 1) {
        const absolute = Math.abs(channelData[sample] ?? 0);
        peak = Math.max(peak, absolute);
        sumSquares += absolute * absolute;
        sampleCount += 1;
      }
    }

    const rms = Math.sqrt(sumSquares / Math.max(1, sampleCount));
    maxPeak = Math.max(maxPeak, peak);
    rawBars.push({ peak, rms });
  }

  const normalizer = maxPeak || 1;
  const bars = rawBars.map((bar) => ({
    peak: Math.max(0.04, Math.min(1, bar.peak / normalizer)),
    rms: Math.max(0.03, Math.min(1, bar.rms / normalizer)),
  }));

  return {
    bars,
    duration: audioBuffer.duration,
    sampleRate: audioBuffer.sampleRate,
    channels: audioBuffer.numberOfChannels,
  };
}

export default App;
