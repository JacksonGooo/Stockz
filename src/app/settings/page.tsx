'use client';

import { useState } from 'react';
import { Layout } from '@/components/layout';
import { Card, Button, Input, Select, Badge } from '@/components/ui';

export default function SettingsPage() {
  const [notifications, setNotifications] = useState(true);
  const [emailAlerts, setEmailAlerts] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [defaultTimeframe, setDefaultTimeframe] = useState('1w');

  return (
    <Layout>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-100 mb-2">
          Settings
        </h1>
        <p className="text-zinc-500 dark:text-zinc-400">
          Customize your experience and preferences
        </p>
      </div>

      <div className="max-w-2xl space-y-6">
        {/* Profile section */}
        <Card>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-6">
            Profile
          </h2>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-bold text-xl">
                U
              </div>
              <div>
                <p className="font-medium text-zinc-900 dark:text-zinc-100">User</p>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">user@example.com</p>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Input label="Display Name" defaultValue="User" />
              <Input label="Email" type="email" defaultValue="user@example.com" />
            </div>
          </div>
        </Card>

        {/* Preferences section */}
        <Card>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-6">
            Preferences
          </h2>
          <div className="space-y-4">
            <Select
              label="Default Prediction Timeframe"
              value={defaultTimeframe}
              onChange={(e) => setDefaultTimeframe(e.target.value)}
              options={[
                { value: '1d', label: '1 Day' },
                { value: '1w', label: '1 Week' },
                { value: '1m', label: '1 Month' },
                { value: '3m', label: '3 Months' },
              ]}
            />

            <ToggleSetting
              label="Dark Mode"
              description="Use dark theme across the application"
              enabled={darkMode}
              onChange={setDarkMode}
            />
          </div>
        </Card>

        {/* Notifications section */}
        <Card>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-6">
            Notifications
          </h2>
          <div className="space-y-4">
            <ToggleSetting
              label="Push Notifications"
              description="Get notified about price alerts and predictions"
              enabled={notifications}
              onChange={setNotifications}
            />
            <ToggleSetting
              label="Email Alerts"
              description="Receive daily market summaries via email"
              enabled={emailAlerts}
              onChange={setEmailAlerts}
            />
          </div>
        </Card>

        {/* AI Model section */}
        <Card>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-6">
            AI Model
          </h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 rounded-xl bg-zinc-50 dark:bg-zinc-800/50">
              <div>
                <p className="font-medium text-zinc-900 dark:text-zinc-100">Model Version</p>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">Currently active prediction model</p>
              </div>
              <Badge variant="info">v0.1.0-stub</Badge>
            </div>
            <div className="flex items-center justify-between p-4 rounded-xl bg-zinc-50 dark:bg-zinc-800/50">
              <div>
                <p className="font-medium text-zinc-900 dark:text-zinc-100">Auto-retrain</p>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">Automatically retrain on new data</p>
              </div>
              <Badge variant="success">Enabled</Badge>
            </div>
          </div>
        </Card>

        {/* Danger zone */}
        <Card className="border-red-200 dark:border-red-900">
          <h2 className="text-lg font-semibold text-red-600 dark:text-red-400 mb-6">
            Danger Zone
          </h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 rounded-xl bg-red-50 dark:bg-red-900/20">
              <div>
                <p className="font-medium text-zinc-900 dark:text-zinc-100">Clear Watchlist</p>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">Remove all stocks from your watchlist</p>
              </div>
              <Button variant="danger" size="sm">Clear</Button>
            </div>
            <div className="flex items-center justify-between p-4 rounded-xl bg-red-50 dark:bg-red-900/20">
              <div>
                <p className="font-medium text-zinc-900 dark:text-zinc-100">Reset Preferences</p>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">Reset all settings to default</p>
              </div>
              <Button variant="danger" size="sm">Reset</Button>
            </div>
          </div>
        </Card>

        {/* Save button */}
        <div className="flex justify-end gap-3">
          <Button variant="ghost">Cancel</Button>
          <Button variant="primary">Save Changes</Button>
        </div>
      </div>
    </Layout>
  );
}

interface ToggleSettingProps {
  label: string;
  description: string;
  enabled: boolean;
  onChange: (enabled: boolean) => void;
}

function ToggleSetting({ label, description, enabled, onChange }: ToggleSettingProps) {
  return (
    <div className="flex items-center justify-between p-4 rounded-xl bg-zinc-50 dark:bg-zinc-800/50">
      <div>
        <p className="font-medium text-zinc-900 dark:text-zinc-100">{label}</p>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">{description}</p>
      </div>
      <button
        onClick={() => onChange(!enabled)}
        className={`
          relative w-12 h-7 rounded-full transition-colors duration-200
          ${enabled ? 'bg-blue-600' : 'bg-zinc-300 dark:bg-zinc-700'}
        `}
      >
        <span
          className={`
            absolute top-1 left-1 w-5 h-5 rounded-full bg-white shadow-sm
            transition-transform duration-200
            ${enabled ? 'translate-x-5' : 'translate-x-0'}
          `}
        />
      </button>
    </div>
  );
}
