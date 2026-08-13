import type { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';

import { Buffer } from 'node:buffer';
import { spawn as spawnChildProcess } from 'node:child_process';
import { StringDecoder } from 'node:string_decoder';

const MAX_STDOUT_BUFFER_BYTES = 4 * 1024 * 1024;
const MAX_STDERR_BUFFER_BYTES = 16 * 1024;

export type JsonLineSidecarErrorCode = 'DISPOSED'
  | 'EXITED'
  | 'PROTOCOL_ERROR'
  | 'REMOTE_ERROR'
  | 'SHUTDOWN_TIMEOUT'
  | 'TIMEOUT'
  | 'UNAVAILABLE';

export interface JsonLineSidecarErrorOptions {
  retryable?: boolean;
  details?: unknown;
  remoteCode?: string;
  cause?: unknown;
}

export class JsonLineSidecarError extends Error {
  readonly code: JsonLineSidecarErrorCode;
  readonly retryable: boolean;
  readonly details: unknown;
  readonly remoteCode: string | undefined;

  constructor(code: JsonLineSidecarErrorCode, message: string, options: JsonLineSidecarErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'JsonLineSidecarError';
    this.code = code;
    this.retryable = options.retryable ?? false;
    this.details = options.details;
    this.remoteCode = options.remoteCode;
  }
}

export interface JsonLineSidecarClientOptions {
  label: string;
  executable: string;
  arguments: string[];
  workingDirectory: string;
  environment: NodeJS.ProcessEnv;
  shutdownTimeoutMs: number;
  spawnProcess?: typeof spawn;
  onStderr?: (output: string) => void;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: JsonLineSidecarError) => void;
  timeout: NodeJS.Timeout;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readRemoteString(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== 'string')
    throw new JsonLineSidecarError('PROTOCOL_ERROR', `Sidecar error.${key} 必须是字符串。`, { details: record });
  return value;
}

export class JsonLineSidecarClient {
  readonly #label: string;
  readonly #executable: string;
  readonly #arguments: string[];
  readonly #workingDirectory: string;
  readonly #environment: NodeJS.ProcessEnv;
  readonly #shutdownTimeoutMs: number;
  readonly #spawnProcess: typeof spawn;
  readonly #onStderr: ((output: string) => void) | undefined;
  readonly #pending = new Map<string, PendingRequest>();

  #child: ChildProcessWithoutNullStreams | undefined;
  #stdoutBuffer = '';
  #stdoutDecoder = new StringDecoder('utf8');
  #stderrDecoder = new StringDecoder('utf8');
  #stderrTail = '';
  #requestSequence = 0;
  #disposing = false;
  #disposed = false;

  constructor(options: JsonLineSidecarClientOptions) {
    this.#label = options.label;
    this.#executable = options.executable;
    this.#arguments = [...options.arguments];
    this.#workingDirectory = options.workingDirectory;
    this.#environment = { ...options.environment };
    this.#shutdownTimeoutMs = options.shutdownTimeoutMs;
    this.#spawnProcess = options.spawnProcess ?? spawnChildProcess;
    this.#onStderr = options.onStderr;
  }

  get disposed(): boolean {
    return this.#disposed;
  }

  request(method: string, params: Record<string, unknown>, timeoutMs: number): Promise<unknown> {
    if (!method.trim())
      throw new JsonLineSidecarError('PROTOCOL_ERROR', 'Sidecar method 不能为空。');
    if (!Number.isInteger(timeoutMs) || timeoutMs <= 0)
      throw new JsonLineSidecarError('PROTOCOL_ERROR', 'Sidecar timeoutMs 必须是正整数。');
    this.#assertAvailable(false);
    const child = this.#ensureProcess();
    const id = `${method}-${++this.#requestSequence}`;

    return new Promise<unknown>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.#pending.delete(id);
        reject(new JsonLineSidecarError('TIMEOUT', `${this.#label} ${method} 请求超时。`, {
          retryable: method !== 'initialize',
          details: { method, timeoutMs },
        }));
      }, timeoutMs);
      this.#pending.set(id, { resolve, reject, timeout });

      try {
        child.stdin.write(`${JSON.stringify({ id, method, params })}\n`, 'utf8', (error) => {
          if (error) {
            this.#rejectPending(id, new JsonLineSidecarError(
              'UNAVAILABLE',
              `无法写入 ${this.#label}。`,
              { cause: error },
            ));
          }
        });
      } catch (error) {
        this.#rejectPending(id, new JsonLineSidecarError(
          'UNAVAILABLE',
          `无法写入 ${this.#label}。`,
          { cause: error },
        ));
      }
    });
  }

  async dispose(): Promise<void> {
    if (this.#disposed)
      return;
    if (this.#disposing)
      throw new JsonLineSidecarError('DISPOSED', `${this.#label} 正在退出。`);

    this.#disposing = true;
    const child = this.#child;
    if (!child) {
      this.#finishDisposal();
      return;
    }

    const deadline = Date.now() + this.#shutdownTimeoutMs;
    try {
      await this.#requestDuringDisposal('shutdown', {}, this.#shutdownTimeoutMs);
      const remainingMs = deadline - Date.now();
      if (remainingMs <= 0)
        throw new JsonLineSidecarError('SHUTDOWN_TIMEOUT', `${this.#label} 未在退出超时内结束。`);
      await this.#waitForExit(child, remainingMs);
      this.#finishDisposal();
    } catch (error) {
      if (child.exitCode === null)
        child.kill('SIGTERM');
      this.#finishDisposal();
      if (error instanceof JsonLineSidecarError && error.code === 'SHUTDOWN_TIMEOUT')
        throw error;
      throw new JsonLineSidecarError(
        'SHUTDOWN_TIMEOUT',
        `${this.#label} 未在退出超时内结束，已终止该客户端创建的子进程。`,
        { cause: error },
      );
    }
  }

  #requestDuringDisposal(
    method: string,
    params: Record<string, unknown>,
    timeoutMs: number,
  ): Promise<unknown> {
    this.#assertAvailable(true);
    const child = this.#ensureProcess();
    const id = `${method}-${++this.#requestSequence}`;
    return new Promise<unknown>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.#pending.delete(id);
        reject(new JsonLineSidecarError('TIMEOUT', `${this.#label} ${method} 请求超时。`));
      }, timeoutMs);
      this.#pending.set(id, { resolve, reject, timeout });
      child.stdin.write(`${JSON.stringify({ id, method, params })}\n`, 'utf8', (error) => {
        if (error)
          this.#rejectPending(id, new JsonLineSidecarError('UNAVAILABLE', `无法写入 ${this.#label}。`, { cause: error }));
      });
    });
  }

  #assertAvailable(allowDisposing: boolean): void {
    if (this.#disposed || (this.#disposing && !allowDisposing))
      throw new JsonLineSidecarError('DISPOSED', `${this.#label} 已退出或正在退出。`);
  }

  #ensureProcess(): ChildProcessWithoutNullStreams {
    if (this.#child && this.#child.exitCode === null)
      return this.#child;

    let child: ReturnType<typeof spawnChildProcess>;
    try {
      child = this.#spawnProcess(this.#executable, this.#arguments, {
        cwd: this.#workingDirectory,
        env: this.#environment,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
    } catch (error) {
      throw new JsonLineSidecarError('UNAVAILABLE', `无法启动 ${this.#label}。`, { cause: error });
    }
    if (!child.stdin || !child.stdout || !child.stderr) {
      child.kill('SIGTERM');
      throw new JsonLineSidecarError('UNAVAILABLE', `${this.#label} 未提供可用的标准输入输出管道。`);
    }

    const pipedChild = child as ChildProcessWithoutNullStreams;
    this.#child = pipedChild;
    this.#stdoutBuffer = '';
    this.#stdoutDecoder = new StringDecoder('utf8');
    this.#stderrDecoder = new StringDecoder('utf8');
    this.#stderrTail = '';
    pipedChild.stdout.on('data', chunk => this.#handleStdout(chunk as Buffer));
    pipedChild.stderr.on('data', chunk => this.#handleStderr(chunk as Buffer));
    pipedChild.stdin.on('error', error => this.#handleProcessFailure(
      pipedChild,
      new JsonLineSidecarError('UNAVAILABLE', `${this.#label} 标准输入不可用。`, { cause: error }),
    ));
    pipedChild.on('error', error => this.#handleProcessFailure(
      pipedChild,
      new JsonLineSidecarError('UNAVAILABLE', `${this.#label} 启动或运行失败。`, { cause: error }),
    ));
    pipedChild.on('close', (code, signal) => this.#handleClose(pipedChild, code, signal));
    return pipedChild;
  }

  #handleStdout(chunk: Buffer): void {
    this.#stdoutBuffer += this.#stdoutDecoder.write(chunk);
    let newlineIndex = this.#stdoutBuffer.indexOf('\n');
    while (newlineIndex >= 0) {
      const line = this.#stdoutBuffer.slice(0, newlineIndex).replace(/\r$/u, '');
      this.#stdoutBuffer = this.#stdoutBuffer.slice(newlineIndex + 1);
      if (!line || Buffer.byteLength(line, 'utf8') > MAX_STDOUT_BUFFER_BYTES) {
        this.#failProtocol(!line ? `${this.#label} 输出了空协议行。` : `${this.#label} 输出行超过协议上限。`);
        return;
      }
      try {
        this.#handleResponse(JSON.parse(line) as unknown);
      } catch (error) {
        if (error instanceof JsonLineSidecarError)
          this.#failProtocol(error.message, error.details);
        else
          this.#failProtocol(`${this.#label} 输出了非法 JSON。`, { line, error });
        return;
      }
      newlineIndex = this.#stdoutBuffer.indexOf('\n');
    }
    if (Buffer.byteLength(this.#stdoutBuffer, 'utf8') > MAX_STDOUT_BUFFER_BYTES)
      this.#failProtocol(`${this.#label} 输出行超过协议上限。`);
  }

  #handleResponse(value: unknown): void {
    if (!isRecord(value))
      throw new JsonLineSidecarError('PROTOCOL_ERROR', `${this.#label} 响应必须是对象。`, { details: value });
    if (typeof value.id !== 'string' || !value.id)
      throw new JsonLineSidecarError('PROTOCOL_ERROR', `${this.#label} 响应 id 必须是非空字符串。`, { details: value });
    const hasResult = Object.hasOwn(value, 'result');
    const hasError = Object.hasOwn(value, 'error');
    if (hasResult === hasError) {
      throw new JsonLineSidecarError(
        'PROTOCOL_ERROR',
        `${this.#label} 响应必须且只能包含 result 或 error。`,
        { details: value },
      );
    }

    const pending = this.#pending.get(value.id);
    if (!pending)
      return;
    this.#pending.delete(value.id);
    clearTimeout(pending.timeout);
    if (hasError) {
      pending.reject(this.#parseRemoteError(value.error));
      return;
    }
    pending.resolve(value.result);
  }

  #parseRemoteError(value: unknown): JsonLineSidecarError {
    if (!isRecord(value))
      throw new JsonLineSidecarError('PROTOCOL_ERROR', `${this.#label} error 必须是对象。`, { details: value });
    const remoteCode = readRemoteString(value, 'code');
    const message = readRemoteString(value, 'message');
    const retryable = value.retryable ?? false;
    if (typeof retryable !== 'boolean') {
      throw new JsonLineSidecarError(
        'PROTOCOL_ERROR',
        `${this.#label} error.retryable 必须是布尔值。`,
        { details: value },
      );
    }
    return new JsonLineSidecarError('REMOTE_ERROR', message, {
      remoteCode,
      retryable,
      details: value.details,
    });
  }

  #handleStderr(chunk: Buffer): void {
    const output = this.#stderrDecoder.write(chunk);
    this.#stderrTail = `${this.#stderrTail}${output}`.slice(-MAX_STDERR_BUFFER_BYTES);
    this.#onStderr?.(output);
  }

  #failProtocol(message: string, details?: unknown): void {
    const error = new JsonLineSidecarError('PROTOCOL_ERROR', message, { details });
    const child = this.#child;
    this.#failAll(error);
    if (child?.exitCode === null)
      child.kill('SIGTERM');
  }

  #handleProcessFailure(child: ChildProcessWithoutNullStreams, error: JsonLineSidecarError): void {
    if (this.#child !== child)
      return;
    this.#failAll(error);
    if (child.exitCode === null)
      child.kill('SIGTERM');
  }

  #handleClose(
    child: ChildProcessWithoutNullStreams,
    code: number | null,
    signal: NodeJS.Signals | null,
  ): void {
    if (this.#child !== child)
      return;
    this.#child = undefined;
    if (this.#disposing)
      return;
    this.#failAll(new JsonLineSidecarError('EXITED', `${this.#label} 意外退出。`, {
      retryable: true,
      details: { code, signal, stderr: this.#stderrTail || undefined },
    }));
  }

  #rejectPending(id: string, error: JsonLineSidecarError): void {
    const pending = this.#pending.get(id);
    if (!pending)
      return;
    this.#pending.delete(id);
    clearTimeout(pending.timeout);
    pending.reject(error);
  }

  #failAll(error: JsonLineSidecarError): void {
    for (const [id, pending] of this.#pending) {
      this.#pending.delete(id);
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
  }

  #finishDisposal(): void {
    this.#failAll(new JsonLineSidecarError('DISPOSED', `${this.#label} 已退出。`));
    this.#disposing = false;
    this.#disposed = true;
    this.#child = undefined;
  }

  #waitForExit(child: ChildProcessWithoutNullStreams, timeoutMs: number): Promise<void> {
    if (this.#child !== child)
      return Promise.resolve();
    return new Promise<void>((resolve, reject) => {
      let timeout: NodeJS.Timeout;
      const onClose = (): void => {
        clearTimeout(timeout);
        resolve();
      };
      timeout = setTimeout(() => {
        child.removeListener('close', onClose);
        reject(new JsonLineSidecarError('SHUTDOWN_TIMEOUT', `${this.#label} 未在退出超时内结束。`));
      }, timeoutMs);
      child.once('close', onClose);
    });
  }
}
