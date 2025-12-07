/**
 * Time formatting utilities with Central Standard Time (CST) support
 */

const CST_TIMEZONE = 'America/Chicago';

/**
 * Format a date to time string in CST
 * @param date - Date object or timestamp
 * @returns Formatted time string with CST indicator (e.g., "2:30:45 PM CST")
 */
export function formatTimeCST(date: Date | number): string {
  const d = typeof date === 'number' ? new Date(date) : date;
  return d.toLocaleTimeString('en-US', {
    timeZone: CST_TIMEZONE,
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  }) + ' CST';
}

/**
 * Format a date to short time string in CST (no seconds)
 * @param date - Date object or timestamp
 * @returns Formatted time string with CST indicator (e.g., "2:30 PM CST")
 */
export function formatTimeShortCST(date: Date | number): string {
  const d = typeof date === 'number' ? new Date(date) : date;
  return d.toLocaleTimeString('en-US', {
    timeZone: CST_TIMEZONE,
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  }) + ' CST';
}

/**
 * Format a date to date and time string in CST
 * @param date - Date object or timestamp
 * @returns Formatted date/time string (e.g., "Dec 4, 2024, 2:30 PM CST")
 */
export function formatDateTimeCST(date: Date | number): string {
  const d = typeof date === 'number' ? new Date(date) : date;
  return d.toLocaleString('en-US', {
    timeZone: CST_TIMEZONE,
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  }) + ' CST';
}

/**
 * Format a timestamp for chart axis labels in CST
 * @param timestamp - Unix timestamp in milliseconds
 * @returns Formatted time string (e.g., "2:30 PM")
 */
export function formatChartTimeCST(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    timeZone: CST_TIMEZONE,
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });
}

/**
 * Get current time in CST as formatted string
 * @returns Current time in CST (e.g., "2:30:45 PM CST")
 */
export function getCurrentTimeCST(): string {
  return formatTimeCST(new Date());
}
