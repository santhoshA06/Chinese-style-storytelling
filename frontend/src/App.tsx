import { Header } from './components/Header'
import { HomePage } from './pages/HomePage'

export default function App() {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(20,184,166,0.22),transparent_34%),radial-gradient(circle_at_top_right,rgba(245,158,11,0.16),transparent_30%),linear-gradient(135deg,#111827_0%,#1c1917_48%,#0f172a_100%)] text-stone-50">
      <Header />
      <HomePage />
    </div>
  )
}
