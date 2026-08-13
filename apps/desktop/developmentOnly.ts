const BLOCKED_FORGE_COMMANDS = new Set(['make', 'package', 'publish']);

export function assertDevelopmentOnlyCommandAllowed(arguments_: readonly string[]): void {
  const blockedCommand = arguments_.find(argument => BLOCKED_FORGE_COMMANDS.has(argument));
  if (blockedCommand) {
    throw new Error(
      `ElectronQwenSpeech 仅支持开发运行，拒绝执行 electron-forge ${blockedCommand}。`,
    );
  }
}
